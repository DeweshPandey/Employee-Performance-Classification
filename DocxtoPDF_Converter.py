import win32com.client
import os


dir_path = "D:\Downloads\Datasets\ResumeClassifier_Piramal\CV train"

file_list = os.listdir(dir_path)
word = win32com.client.Dispatch("Word.Application")
word.Visible  = False
not_converted = []

for file_name in file_list:
    
    docx_path = f"{dir_path}\{file_name}"
    pdf_path = f"{dir_path[:-8]}CV_train_pdf\{file_name[:-4]}pdf"
    
    if f"{file_name[:-4]}pdf" in os.listdir(f"{dir_path[:-8]}CV_train_pdf"):
        print(f"{file_name[:-4]}pdf already converted ")
        continue
    print(pdf_path)
    
    
    def docx_to_pdf(docx_path, pdf_path):
        try : 
        
            doc = word.Documents.Open(docx_path)
            doc.SaveAs(pdf_path, FileFormat=17)  # 17 corresponds to PDF format
            doc.Close()
            print(f"PDF saved: {pdf_path}")
            
        except Exception  as e:
            # Add the file to the not_converted list
            not_converted.append(file_name)
            print(f"Failed to convert {docx_path}: {e}")
    
    docx_to_pdf(docx_path, pdf_path)
      
print(not_converted)
word.Quit()

# # Usage
        


# from spire.doc import *
# from spire.doc.common import *
        
# # Create a Document object
# document = Document()
# # Load a Word DOCX file
# document.LoadFromFile(docx_path)
# # Or load a Word DOC file
# #document.LoadFromFile("Sample.doc")

# # Save the file to a PDF file
# document.SaveToFile(pdf_path, FileFormat.PDF)
# document.Close()