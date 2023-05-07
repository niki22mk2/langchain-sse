from bs4 import BeautifulSoup, Tag, NavigableString
from bs4.element import Comment
from readability import Document
from pyppeteer import launch
from pyppeteer.browser import Browser
from pyppeteer.page import Page
from extractcontent3 import ExtractContent
import asyncio, re

async def _get_page(url):
    browser: Browser = await launch(
        headless=True, handleSIGINT=False, handleSIGTERM=False, handleSIGHUP=False
    )
    page: Page = await browser.newPage()
    try:
        await page.goto(url)
        await page.setViewport({"width": 1920, "height": 1080})
        content = await page.content()
    except Exception as e:
        await browser.close()
        raise Exception("Get Page Error:", e)
    await browser.close()
    return content

def _extract_content_readability(html):
    doc = Document(html)

    title = doc.short_title()
    content_html = doc.summary()

    soup = BeautifulSoup(content_html, 'html.parser')
    for element in soup(
        [
            "title",
        ]
    ):
        element.extract()    

    content_text = soup.get_text().strip()

    return title,f"{title}\n\n{content_text}"


def _extract_content_extractcontent3(html):
    extractor = ExtractContent()
    opt = {"threshold":50}
    extractor.set_option(opt)

    extractor.analyse(html)

    content_text, title = extractor.as_text()

    content_text = content_text.strip()
    
    return title,f"{title}\n\n{content_text}"


def _extract_content_manual(html, max_len=30000):
    try:
        soup = BeautifulSoup(html, "html.parser")

        content = soup.main
        if content is None:
            content = soup.body

        for element in soup(
            [
                "style",
                "head",
                "meta",
                "[document]",
                "object",
                "input",
                "form",
                "textarea",
                "button",
                "noscript",
                "fecolormatrix",
                "filter",
                "svg",
                "aside",
                "script",
                "footer",
                "nav"
            ]
        ):
            element.extract()

        for element in content(text=lambda text: isinstance(text, Comment)):
            element.extract()

        # class名を含む要素を除外
        for element in content.find_all(lambda tag: tag.has_attr('class') and any(cls_name.lower() in cls.lower() for cls_name in ['side', 'header', 'adv','footer','modal','alert','inner','navi','tab','sub','reaction'] for cls in tag['class'])):
            element.extract()

        # id名を含む要素を除外
        for element in content.find_all(lambda tag: tag.has_attr('id') and any(id_name.lower() in id.lower() for id_name in ['side', 'header', 'adv','footer','modal','alert','inner','navi','tab','sub','reaction'] for id in tag['id'])):
            element.extract()

        def process_node(node):
            if isinstance(node, NavigableString):
                return node.strip()
            elif isinstance(node, Tag):
                if node.name in ['br', 'p', 'li', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    return '\n' + ''.join(process_node(child) for child in node.children)
                else:
                    return ''.join(process_node(child) for child in node.children)
            else:
                return ''

        text = process_node(content).strip()
        text = re.sub(r'(\s*\n\s*){3,}', '\n', text)

    except:
        raise Exception("HTML Parse Error.")

    if len(text) > max_len:
        text = text[:max_len]
        text += "\n注意:本文は最大長を超えたため、切り捨てられています。"

    return text

async def extract_content(url):
    html = await _get_page(url)

    titl11,content1 = _extract_content_readability(html)
    titl12,content2 = _extract_content_extractcontent3(html)
    content3 = _extract_content_manual(html)

    if (len(content1) + len(content2)) <= len(content3)/5:
        print("extract_content_manual")
        content_text = content3
        title = titl11
    elif len(content1) >= len(content2):
        print("extract_content_readability")
        content_text = content1
        title = titl11
    elif len(content1) <= len(content2):
        print("extract_content_extractcontent3")
        content_text = content2
        title = titl12

    if title is None or title == "":
        title = "No Title Webpage"
    
    if len(content_text) > 80000:
        content_text = content_text[:80000]
        content_text += "\n注意:本文は最大長を超えたため、切り捨てられています。"

    return title,f"{title}\n\n{content_text}"

