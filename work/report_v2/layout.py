"""Research-report layout. Narrative is written separately from raw results."""
from pathlib import Path
import re
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4
from PIL import Image as PILImage

ROOT=Path(__file__).resolve().parent
pdfmetrics.registerFont(TTFont('CN','C:/Windows/Fonts/msyh.ttc'))
pdfmetrics.registerFont(TTFont('CNB','C:/Windows/Fonts/msyhbd.ttc'))
pdfmetrics.registerFont(TTFont('MATH','C:/Windows/Fonts/cambria.ttc'))
pdfmetrics.registerFontFamily('CN',normal='CN',bold='CNB',italic='CN',boldItalic='CNB')
ST={
 'body':ParagraphStyle('body',fontName='CN',fontSize=10.1,leading=16.5,spaceAfter=8,wordWrap='CJK'),
 'small':ParagraphStyle('small',fontName='CN',fontSize=8.4,leading=12.7,spaceAfter=6,wordWrap='CJK'),
 'title':ParagraphStyle('title',fontName='CNB',fontSize=23,leading=33,spaceAfter=18,wordWrap='CJK'),
 'head':ParagraphStyle('head',fontName='CNB',fontSize=15,leading=23,spaceBefore=3,spaceAfter=12,wordWrap='CJK',keepWithNext=True),
 'sub':ParagraphStyle('sub',fontName='CNB',fontSize=11.4,leading=18,spaceAfter=7,spaceBefore=4,wordWrap='CJK',keepWithNext=True),
 'cell':ParagraphStyle('cell',fontName='CN',fontSize=8.5,leading=12.8,wordWrap='CJK'),
 'cap':ParagraphStyle('cap',fontName='CN',fontSize=8.3,leading=12.4,spaceBefore=5,spaceAfter=9,wordWrap='CJK')}
story=[];transcript=[]
def glyphs(t):
    sup=str.maketrans('⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺','0123456789-+')
    sub=str.maketrans('₀₁₂₃₄₅₆₇₈₉₋₊','0123456789-+')
    t=re.sub('[⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺]+',lambda m:'<super>'+m.group().translate(sup)+'</super>',t)
    t=re.sub('[₀₁₂₃₄₅₆₇₈₉₋₊]+',lambda m:'<sub>'+m.group().translate(sub)+'</sub>',t)
    return t.replace('∇','<font name="MATH">∇</font>')
def p(text,style='body'):
    story.append(Paragraph(glyphs(text),ST[style]));transcript.append(text)
def heading(text):p(text,'head')
def sub(text):p(text,'sub')
def new(text):story.append(PageBreak());heading(text)
def table(headers,rows,widths):
    cells=[[Paragraph(glyphs(str(x)),ST['cell']) for x in row] for row in [headers]+rows]
    tab=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
    tab.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E6EEF4')),('GRID',(0,0),(-1,-1),.4,colors.HexColor('#D0D8DE')),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
    story.extend([tab,Spacer(1,8)]);transcript.extend([' | '.join(map(str,r)) for r in [headers]+rows])
def fig(name,width=490,caption=None):
    path=ROOT/'figures'/name
    with PILImage.open(path) as im:w,h=im.size
    story.append(Image(str(path),width=width,height=width*h/w))
    if caption:p(caption,'cap')
def eq(name):
    with PILImage.open(ROOT/'figures'/(name+'.png')) as im:w=im.width
    fig(name+'.png',width=min(490,w*72/220*.88))
def footer(canvas,doc):
    canvas.saveState();canvas.setStrokeColor(colors.HexColor('#CFD8DF'));canvas.line(49,43,546,43)
    canvas.setFont('CN',8);canvas.setFillColor(colors.HexColor('#536575'))
    canvas.drawString(50,29,'不可压 NS 神经计算研究  |  第二轮验证  |  2026-11-09')
    canvas.drawRightString(545,29,str(doc.page));canvas.restoreState()
def build(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    doc=SimpleDocTemplate(str(path),pagesize=A4,rightMargin=49,leftMargin=49,topMargin=45,bottomMargin=57,
        title='不可压 NS 神经计算研究报告：复杂算例与公平比较',author='Computational research study')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    (ROOT/'report_text.txt').write_text('\n\n'.join(transcript),encoding='utf8')
