from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)

system_prompt_template = "{system}"

human_message_template = "".join([
    "{relevant}\n",
    "{information}\n\n",
    "ユーザのメッセージ:\n",
    "{input}\n\n",
    "発言の例やルールを守ったあなたの返答:"
])

PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_prompt_template),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template(human_message_template)
])