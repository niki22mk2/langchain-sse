from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain import PromptTemplate

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


template = """
会話履歴から、{name}が長期的に記憶すべき内容を要約してください。
要約の最初にいつ頃の出来事かを記載してください。複数の話題が含まれている場合は、複数の要約を作成してください。
なお、{name}はあなた自身であるため、一人称で表現してください。

フォーマット:
xxxx年x月x日 x時頃、～～話題1の要約の文章～～
yyyy年y月y日 y時頃、～～話題2の要約の文章～～

会話履歴:
{history}

フォーマットに従った要約:
"""
MEMORY_PROMPT = PromptTemplate(template=template, input_variables=["name", "history"])