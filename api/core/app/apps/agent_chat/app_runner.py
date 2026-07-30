import logging
from typing import cast

from core.agent.cot_chat_agent_runner import CotChatAgentRunner
from core.agent.cot_completion_agent_runner import CotCompletionAgentRunner
from core.agent.entities import AgentEntity
from core.agent.fc_agent_runner import FunctionCallAgentRunner
from core.app.apps.agent_chat.app_config_manager import AgentChatAppConfig
from core.app.apps.base_app_queue_manager import AppQueueManager, PublishFrom
from core.app.apps.base_app_runner import AppRunner
from core.app.entities.app_invoke_entities import AgentChatAppGenerateEntity
from core.app.entities.queue_entities import QueueAnnotationReplyEvent
from core.memory.token_buffer_memory import TokenBufferMemory
from core.model_manager import ModelInstance
from core.model_runtime.entities.llm_entities import LLMMode
from core.model_runtime.entities.model_entities import ModelFeature, ModelPropertyKey
from core.model_runtime.model_providers.__base.large_language_model import LargeLanguageModel
from core.moderation.base import ModerationError
from extensions.ext_database import db
from models.model import App, Conversation, Message

logger = logging.getLogger(__name__)


class AgentChatAppRunner(AppRunner):
    """
    Agent Application Runner
    """

    def run(
        self,
        application_generate_entity: AgentChatAppGenerateEntity,
        queue_manager: AppQueueManager,
        conversation: Conversation,
        message: Message,
    ) -> None:
        """
        Run assistant application
        :param application_generate_entity: application generate entity
        :param queue_manager: application queue manager
        :param conversation: conversation
        :param message: message
        :return:
        """
        app_config = application_generate_entity.app_config
        app_config = cast(AgentChatAppConfig, app_config)

        # 根据app_id查询apps表，看app是否存在
        app_record = db.session.query(App).filter(App.id == app_config.app_id).first()
        if not app_record:
            raise ValueError("App not found")

        inputs = application_generate_entity.inputs
        query = application_generate_entity.query
        files = application_generate_entity.files

        # Pre-calculate the number of tokens of the prompt messages,
        # and return the rest number of tokens by model context token size limit and max token size limit.
        # If the rest number of tokens is not enough, raise exception.
        # Include: prompt template, inputs, query(optional), files(optional)
        # Not Include: memory, external data, dataset context
        # 根据 prompt template, inputs, files、query（不包括memory、external data、dataset context）计算token，得到剩余可用token。如果剩余可用token不足则抛异常
        self.get_pre_calculate_rest_tokens(
            app_record=app_record,
            model_config=application_generate_entity.model_conf,
            prompt_template_entity=app_config.prompt_template,
            inputs=dict(inputs),
            files=list(files),
            query=query,
        )

        # 读取历史对话信息（如果存在）
        memory = None
        if application_generate_entity.conversation_id:
            # get memory of conversation (read-only)
            model_instance = ModelInstance(
                provider_model_bundle=application_generate_entity.model_conf.provider_model_bundle,
                model=application_generate_entity.model_conf.model,
            )

            memory = TokenBufferMemory(conversation=conversation, model_instance=model_instance)

        # organize all inputs and template to prompt messages  生成提示消息
        # Include: prompt template, inputs, query(optional), files(optional)
        #          memory(optional)
        prompt_messages, _ = self.organize_prompt_messages(
            app_record=app_record,
            model_config=application_generate_entity.model_conf,
            prompt_template_entity=app_config.prompt_template,
            inputs=dict(inputs),
            files=list(files),
            query=query,
            memory=memory,
        )

        # moderation
        try:
            # process sensitive_word_avoidance  检查用户输入的文本是否合规（例如是否包含敏感、暴力、违法或违反平台政策的内容）
            _, inputs, query = self.moderation_for_inputs(
                app_id=app_record.id,
                tenant_id=app_config.tenant_id,
                app_generate_entity=application_generate_entity,
                inputs=dict(inputs),
                query=query or "",
                message_id=message.id,
            )
        except ModerationError as e:
            self.direct_output(
                queue_manager=queue_manager,
                app_generate_entity=application_generate_entity,
                prompt_messages=prompt_messages,
                text=str(e),
                stream=application_generate_entity.stream,
            )
            return

        if query:
            # annotation reply  如果开启标注回复，则通过AnnotationReplyFeature从数据库中查询答案，并跳到handle response那一步
            annotation_reply = self.query_app_annotations_to_reply(
                app_record=app_record,
                message=message,
                query=query,
                user_id=application_generate_entity.user_id,
                invoke_from=application_generate_entity.invoke_from,
            )

            if annotation_reply:
                queue_manager.publish(
                    QueueAnnotationReplyEvent(message_annotation_id=annotation_reply.id),
                    PublishFrom.APPLICATION_MANAGER,
                )

                self.direct_output(
                    queue_manager=queue_manager,
                    app_generate_entity=application_generate_entity,
                    prompt_messages=prompt_messages,
                    text=annotation_reply.content,
                    stream=application_generate_entity.stream,
                )
                return

        # fill in variable inputs from external data tools if exists 如果存在外部数据工具，则从其中填充变量输入
        external_data_tools = app_config.external_data_variables
        if external_data_tools:
            inputs = self.fill_in_inputs_from_external_data_tools(
                tenant_id=app_record.tenant_id,
                app_id=app_record.id,
                external_data_tools=external_data_tools,
                inputs=inputs,
                query=query,
            )

        # reorganize all inputs and template to prompt messages
        # Include: prompt template, inputs, query(optional), files(optional)
        #          memory(optional), external data, dataset context(optional)
        prompt_messages, _ = self.organize_prompt_messages(
            app_record=app_record,
            model_config=application_generate_entity.model_conf,
            prompt_template_entity=app_config.prompt_template,
            inputs=dict(inputs),
            files=list(files),
            query=query or "",
            memory=memory,
        )

        # check hosting moderation
        hosting_moderation_result = self.check_hosting_moderation(
            application_generate_entity=application_generate_entity,
            queue_manager=queue_manager,
            prompt_messages=prompt_messages,
        )

        if hosting_moderation_result:
            return

        agent_entity = app_config.agent
        assert agent_entity is not None

        # init model instance  初始化模型实例
        model_instance = ModelInstance(
            provider_model_bundle=application_generate_entity.model_conf.provider_model_bundle,
            model=application_generate_entity.model_conf.model,
        )
        prompt_message, _ = self.organize_prompt_messages(
            app_record=app_record,
            model_config=application_generate_entity.model_conf,
            prompt_template_entity=app_config.prompt_template,
            inputs=dict(inputs),
            files=list(files),
            query=query or "",
            memory=memory,
        )

        # change function call strategy based on LLM model   根据 LLM 模型更改函数调用策略
        llm_model = cast(LargeLanguageModel, model_instance.model_type_instance)
        model_schema = llm_model.get_model_schema(model_instance.model, model_instance.credentials)
        if not model_schema:
            raise ValueError("Model schema not found")

        if {ModelFeature.MULTI_TOOL_CALL, ModelFeature.TOOL_CALL}.intersection(model_schema.features or []):
            agent_entity.strategy = AgentEntity.Strategy.FUNCTION_CALLING

        conversation_result = db.session.query(Conversation).filter(Conversation.id == conversation.id).first()
        if conversation_result is None:
            raise ValueError("Conversation not found")
        message_result = db.session.query(Message).filter(Message.id == message.id).first()
        if message_result is None:
            raise ValueError("Message not found")
        db.session.close()

        # 根据策略选择不同的runner
        runner_cls: type[FunctionCallAgentRunner] | type[CotChatAgentRunner] | type[CotCompletionAgentRunner]
        # start agent runner
        if agent_entity.strategy == AgentEntity.Strategy.CHAIN_OF_THOUGHT:
            # check LLM mode
            if model_schema.model_properties.get(ModelPropertyKey.MODE) == LLMMode.CHAT.value:
                runner_cls = CotChatAgentRunner
            elif model_schema.model_properties.get(ModelPropertyKey.MODE) == LLMMode.COMPLETION.value:
                runner_cls = CotCompletionAgentRunner
            else:
                raise ValueError(f"Invalid LLM mode: {model_schema.model_properties.get(ModelPropertyKey.MODE)}")
        elif agent_entity.strategy == AgentEntity.Strategy.FUNCTION_CALLING:
            runner_cls = FunctionCallAgentRunner
        else:
            raise ValueError(f"Invalid agent strategy: {agent_entity.strategy}")

        runner = runner_cls(
            tenant_id=app_config.tenant_id,
            application_generate_entity=application_generate_entity,
            conversation=conversation_result,
            app_config=app_config,
            model_config=application_generate_entity.model_conf,
            config=agent_entity,
            queue_manager=queue_manager,
            message=message_result,
            user_id=application_generate_entity.user_id,
            memory=memory,
            prompt_messages=prompt_message,
            model_instance=model_instance,
        )

        # 运行runner
        invoke_result = runner.run(
            message=message,
            query=query,
            inputs=inputs,
        )

        # handle invoke result  处理调用结果
        self._handle_invoke_result(
            invoke_result=invoke_result,
            queue_manager=queue_manager,
            stream=application_generate_entity.stream,
            agent=True,
        )
