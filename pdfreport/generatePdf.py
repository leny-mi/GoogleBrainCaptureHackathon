from fpdf import FPDF
import os
import datetime

class Report(FPDF):
    def __init__(self, artifacts:list, annotationInfo:dict,):
        super().__init__()
        self.artifacts = artifacts
        self.annotationInfo = annotationInfo
        self._assertAnnotationInfo()

        self.leftMostX = 10
        self.fontFamily = "Arial"

    def _assertAnnotationInfo(self,):
        assert 'annotater' in self.annotationInfo, "You have to supply annotater"
        assert 'annotationDate' in self.annotationInfo, "You have to supply annotation date"
    
    def _logo(self, x, y, w, h, ):
        self.image(name = "pdfreport/images/BCLogo.png", x = x, y = y, w = w, h = h, link = 'https://braincapture.dk')

    def _title(self, y, fontSize):
        self.set_font(self.fontFamily, 'B', fontSize)
        self.set_xy(x = 0, y = y)
        title = "EEG Artifacts Report"
        self.cell(w=self.w, txt = title, align = 'C')

    def _annotationMetaData(self, verticalSpace:int, lineSpacing:int):
        firstColumnX = self.leftMostX
        self.set_font_size(12)
        self.set_xy(firstColumnX, self.y + verticalSpace)
        self.text(firstColumnX, self.y, F"Annotater: {self.annotationInfo['annotater']}")
        self.text(firstColumnX, self.y+lineSpacing, F"Date of annotation: {self.annotationInfo['annotationDate'].isoformat()}")

    def _displayArtifact(self, type:str, start:str, duration:str):
        xType = self.leftMostX
        xStart = 75
        xDuration = 150
        self.set_x(xType)
        self.text(self.x, self.y, type)
        self.set_x(xStart)
        self.text(self.x, self.y, start)
        self.set_x(xDuration)
        self.text(self.x, self.y, duration)

    def _displayArtifacts(self, verticalSpace:int, lineSpacing:int, lines:bool = True):
        self.set_xy(self.leftMostX, self.y + verticalSpace)
        self.set_font(self.fontFamily, 'B', 12)
        self._displayArtifact("Artifact Type", "Start of Artifact", "Duration of Artifact") # Table Header
        self.set_font(self.fontFamily, '', 12)
        self.set_xy(self.leftMostX, self.y + lineSpacing)
        for artifact in self.artifacts:
            self._displayArtifact(artifact[0], artifact[1], artifact[2])
            self.set_y(self.y + lineSpacing)
            


    def main(self, ):
        self.add_page()
        self._logo(x= 5, y = 5, w = 50, h = 25)
        self._title(y = 40, fontSize = 25)
        self._annotationMetaData(verticalSpace = 20, lineSpacing = 6,)
        self._displayArtifacts(verticalSpace=20, lineSpacing=6)

    def __call__(self, name:str, dest:int= ''):
        assert isinstance(name, str) and name[-4:] == ".pdf", "Name has to be string and and with '.pdf'"
        self.main()
        self.output(name = name, dest = dest)

if __name__ == '__main__':
    artifacts = [
        ("Eye", "3", "10"),
        ("Blink", "10", "2"),
        ("Something else", "15", "3")
    ]
    annotationInfo = dict(
        annotater="Doctor Gruny", 
        annotationDate=datetime.date(2024, 3, 13)
    )
    report = Report(artifacts=artifacts, annotationInfo=annotationInfo)
    report(name = "Report.pdf")