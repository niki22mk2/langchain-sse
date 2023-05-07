from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)

system_prompt_template = "{system}"

human_message_template = """
{date}

（参考情報）関連するかもしれない過去の会話:
{relevant}

（参考情報）その他情報:
{information}

ユーザのメッセージ:
{input}

会話のルールに従ったあなたの返答:"""

PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_prompt_template),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template(human_message_template)
])