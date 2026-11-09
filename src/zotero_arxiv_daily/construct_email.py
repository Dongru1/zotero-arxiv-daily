from .protocol import Paper
import math


framework = """
<!DOCTYPE HTML>
<html>
<head>
  <style>
    .star-wrapper {
      font-size: 1.3em; /* 调整星星大小 */
      line-height: 1; /* 确保垂直对齐 */
      display: inline-flex;
      align-items: center; /* 保持对齐 */
    }
    .half-star {
      display: inline-block;
      width: 0.5em; /* 半颗星的宽度 */
      overflow: hidden;
      white-space: nowrap;
      vertical-align: middle;
    }
    .full-star {
      vertical-align: middle;
    }
  </style>
</head>
<body>

<div>
    __CONTENT__
</div>

<br><br>
<div>
To unsubscribe, remove your email in your Github Action setting.
</div>

</body>
</html>
"""

def get_empty_html():
  block_template = """
  <table border="0" cellpadding="0" cellspacing="0" width="100%" style="font-family: Arial, sans-serif; border: 1px solid #ddd; border-radius: 8px; padding: 16px; background-color: #f9f9f9;">
  <tr>
    <td style="font-size: 20px; font-weight: bold; color: #333;">
        No Papers Today. Take a Rest!
    </td>
  </tr>
  </table>
  """
  return block_template

def _page_links(paper: Paper) -> list[tuple[str, str]]:
    """Buttons for the abstract or forum page, separate from the PDF."""
    links: list[tuple[str, str]] = []
    seen: set[str] = set()

    def add(label: str, url: str | None) -> None:
        if not url or url in seen or url == paper.pdf_url:
            return
        seen.add(url)
        links.append((label, url))

    url = paper.url or ""
    if "openreview.net" in url:
        add("OpenReview", url)
    arxiv_url = paper.arxiv_url or ""
    if not arxiv_url and "arxiv.org/abs/" in url:
        arxiv_url = url
    if not arxiv_url and paper.pdf_url and "arxiv.org/pdf/" in paper.pdf_url:
        arxiv_url = paper.pdf_url.replace("/pdf/", "/abs/", 1)
    add("arXiv", arxiv_url)
    if url and "openreview.net" not in url and "arxiv.org" not in url:
        add("Page", url)
    return links


def get_block_html(title:str, authors:str, rate:str, tldr:str, pdf_url:str, affiliations:str=None, page_links:list[tuple[str, str]] | None = None):
    buttons = []
    for label, url in page_links or []:
        buttons.append(
            f'<a href="{url}" style="display: inline-block; text-decoration: none; font-size: 14px; font-weight: bold; color: #fff; background-color: #0275d8; padding: 8px 16px; border-radius: 4px; margin-right: 8px;">{label}</a>'
        )
    if pdf_url:
        buttons.append(
            f'<a href="{pdf_url}" style="display: inline-block; text-decoration: none; font-size: 14px; font-weight: bold; color: #fff; background-color: #d9534f; padding: 8px 16px; border-radius: 4px;">PDF</a>'
        )
    title_html = title
    if page_links:
        title_html = f'<a href="{page_links[0][1]}" style="color: #333; text-decoration: none;">{title}</a>'
    block_template = """
    <table border="0" cellpadding="0" cellspacing="0" width="100%" style="font-family: Arial, sans-serif; border: 1px solid #ddd; border-radius: 8px; padding: 16px; background-color: #f9f9f9;">
    <tr>
        <td style="font-size: 20px; font-weight: bold; color: #333;">
            {title}
        </td>
    </tr>
    <tr>
        <td style="font-size: 14px; color: #666; padding: 8px 0;">
            {authors}
            <br>
            <i>{affiliations}</i>
        </td>
    </tr>
    <tr>
        <td style="font-size: 14px; color: #333; padding: 8px 0;">
            <strong>Relevance:</strong> {rate}
        </td>
    </tr>
    <tr>
        <td style="font-size: 14px; color: #333; padding: 8px 0;">
            <strong>TLDR:</strong> {tldr}
        </td>
    </tr>

    <tr>
        <td style="padding: 8px 0;">
            {buttons}
        </td>
    </tr>
</table>
"""
    return block_template.format(
        title=title_html,
        authors=authors,
        rate=rate,
        tldr=tldr,
        buttons="".join(buttons),
        affiliations=affiliations,
    )

def get_stars(score:float):
    full_star = '<span class="full-star">⭐</span>'
    half_star = '<span class="half-star">⭐</span>'
    low = 6
    high = 8
    if score <= low:
        return ''
    elif score >= high:
        return full_star * 5
    else:
        interval = (high-low) / 10
        star_num = math.ceil((score-low) / interval)
        full_star_num = int(star_num/2)
        half_star_num = star_num - full_star_num * 2
        return '<div class="star-wrapper">'+full_star * full_star_num + half_star * half_star_num + '</div>'


def render_email(papers:list[Paper], intro:str | None = None) -> str:
    parts = []
    if len(papers) == 0 :
        body = get_empty_html()
        if intro:
            body = f'<p style="font-family: Arial, sans-serif; font-size: 15px; color: #333;">{intro}</p>' + body
        return framework.replace('__CONTENT__', body)
    
    for p in papers:
        #rate = get_stars(p.score)
        rate = round(p.score, 1) if p.score is not None else 'Unknown'
        author_list = [a for a in p.authors]
        num_authors = len(author_list)
        if num_authors <= 5:
            authors = ', '.join(author_list)
        else:
            authors = ', '.join(author_list[:3] + ['...'] + author_list[-2:])
        if p.affiliations is not None:
            affiliations = p.affiliations[:5]
            affiliations = ', '.join(affiliations)
            if len(p.affiliations) > 5:
                affiliations += ', ...'
        else:
            affiliations = 'Unknown Affiliation'
        parts.append(get_block_html(p.title, authors, rate, p.tldr, p.pdf_url, affiliations, _page_links(p)))

    content = '<br>' + '</br><br>'.join(parts) + '</br>'
    if intro:
        content = (
            f'<p style="font-family: Arial, sans-serif; font-size: 15px; color: #333;">{intro}</p>'
            + content
        )
    return framework.replace('__CONTENT__', content)
