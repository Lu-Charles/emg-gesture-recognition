"""Typeset the verified Markdown manuscript as a readable research PDF."""
import re,html
from pathlib import Path
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,KeepTogether,PageBreak
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.colors import HexColor,white
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_LEFT
from PIL import Image as PILImage
R=Path(__file__).resolve().parents[1];O=R/'research/expanded_study_20260916';P=R/'output/pdf';P.mkdir(parents=True,exist_ok=True)
fonts=Path('~/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/libreoffice-headless/libreoffice/LibreOfficeDev.app/Contents/Resources/fonts/truetype')
for suffix,name in [('Regular','Serif'),('Bold','SerifBold'),('Italic','SerifItalic')]:pdfmetrics.registerFont(TTFont(name,str(fonts/f'LiberationSerif-{suffix}.ttf')))
pdfmetrics.registerFontFamily('Serif',normal='Serif',bold='SerifBold',italic='SerifItalic',boldItalic='SerifBold')
pdfmetrics.registerFont(TTFont('Sans',str(fonts/'DejaVuSans.ttf')))
styles={
 'body':ParagraphStyle('Body',fontName='Serif',fontSize=10.5,leading=14.3,spaceAfter=7,splitLongWords=True),
 'title':ParagraphStyle('Title',fontName='SerifBold',fontSize=22,leading=25,spaceAfter=9,textColor=HexColor('#173849')),
 'subtitle':ParagraphStyle('Sub',fontName='Serif',fontSize=13,leading=16,spaceAfter=12),
 'h2':ParagraphStyle('H2',fontName='SerifBold',fontSize=13,leading=16,spaceBefore=13,spaceAfter=7,keepWithNext=True),
 'h3':ParagraphStyle('H3',fontName='SerifBold',fontSize=11.3,leading=14,spaceBefore=10,spaceAfter=6,keepWithNext=True),
 'caption':ParagraphStyle('Cap',fontName='Serif',fontSize=9.1,leading=11.8,spaceAfter=11),
 'cell':ParagraphStyle('Cell',fontName='Serif',fontSize=8.8,leading=11.5),
 'headcell':ParagraphStyle('HeadCell',fontName='SerifBold',fontSize=8.8,leading=11.5,textColor=white),
 'reference':ParagraphStyle('Ref',fontName='Serif',fontSize=9,leading=12,spaceAfter=7,splitLongWords=True),
}
def para(text,style='body'):
 text=html.escape(text);text=re.sub(r'(https?://[^\s]+)',r'<link href="\1" color="#0c637a">\1</link>',text);return Paragraph(text,styles[style])
lines=(O/'Expanded_EMG_Manuscript.md').read_text().splitlines();story=[];i=0;firstsub=True;references=False
while i<len(lines):
 line=lines[i].strip()
 if not line:i+=1;continue
 if line.startswith('# '):story.append(para(line[2:],'title'));i+=1;continue
 if line.startswith('## '):
  text=line[3:];style='subtitle' if firstsub else 'h2';firstsub=False;references=text=='References';
  if references:story.append(PageBreak())
  story.append(para(text,style));i+=1;continue
 if line.startswith('### '):story.append(para(line[4:],'h3'));i+=1;continue
 if line.startswith('|'):
  rows=[]
  while i<len(lines) and lines[i].strip().startswith('|'):
   row=[v.strip() for v in lines[i].strip().strip('|').split('|')]
   if not all(re.fullmatch(r'[:\- ]+',v) for v in row):rows.append(row)
   i+=1
  n=len(rows[0]);width=512;first=166 if n<=4 else 139;widths=[first]+[(width-first)/(n-1)]*(n-1)
  cells=[[para(v,'headcell' if k==0 else 'cell') for v in row] for k,row in enumerate(rows)];table=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
  table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),HexColor('#254c5d')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),('ROWBACKGROUNDS',(0,1),(-1,-1),[HexColor('#f1f5f6'),white]),('LINEBELOW',(0,-1),(-1,-1),.5,HexColor('#9cafb8'))]))
  j=i
  while j<len(lines) and not lines[j].strip():j+=1
  if j<len(lines) and lines[j].startswith('Table'):
   story.append(KeepTogether([table,Spacer(1,5),para(lines[j],'caption')]));i=j+1
  else:story.extend([table,Spacer(1,9)])
  continue
 if line.startswith('!['):
  path=O/re.search(r'\]\((.+)\)',line)[1];im=PILImage.open(path);h=512*im.height/im.width;group=[Image(str(path),width=512,height=h),Spacer(1,4)];j=i+1
  while j<len(lines) and not lines[j].strip():j+=1
  if j<len(lines) and lines[j].startswith('Figure'):group.append(para(lines[j],'caption'));i=j+1
  else:i+=1
  story.append(KeepTogether(group));continue
 paragraph=[line];i+=1
 while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|','![')):paragraph.append(lines[i].strip());i+=1
 story.append(para(' '.join(paragraph),'reference' if references else 'body'))
# Merge headings with table/image groups to prevent a stranded heading.
fixed=[];j=0
while j<len(story):
 if isinstance(story[j],Paragraph) and getattr(story[j].style,'keepWithNext',False) and j+1<len(story) and isinstance(story[j+1],KeepTogether):
  fixed.append(KeepTogether([story[j],*story[j+1]._content]));j+=2
 else:fixed.append(story[j]);j+=1
story=fixed
def page(canvas,doc):
 canvas.saveState();canvas.setFont('Sans',7.3);canvas.setFillColor(HexColor('#687e87'));canvas.drawString(50,764,'BRIEF EMG CALIBRATION | Research manuscript draft');canvas.drawRightString(562,764,'Charles Lu');canvas.setStrokeColor(HexColor('#c3d0d5'));canvas.line(50,754,562,754);canvas.drawString(50,27,'EMG calibration evaluation');canvas.drawRightString(562,27,str(doc.page));canvas.restoreState()
doc=SimpleDocTemplate(str(P/'Charles_Lu_Expanded_EMG_Study.pdf'),pagesize=(612,792),leftMargin=50,rightMargin=50,topMargin=51,bottomMargin=48,title='Brief EMG Calibration Does Not Guarantee Vocabulary Preservation',author='Charles Lu')
doc.build(story,onFirstPage=page,onLaterPages=page)
print(P/'Charles_Lu_Expanded_EMG_Study.pdf')
