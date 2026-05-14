"""
多格式文档读取器
支持：TXT, PDF, DOCX, DOC, MD, CSV, JSON 等格式
"""
import os
from typing import Optional


class DocumentReader:
    """多格式文档读取器"""
    
    # 支持的文本格式
    TEXT_EXTENSIONS = {'.txt', '.md', '.csv', '.json', '.xml', '.html', '.htm', '.log', '.rtf'}
    # 支持的 PDF 格式
    PDF_EXTENSIONS = {'.pdf'}
    # 支持的 Word 格式
    WORD_EXTENSIONS = {'.docx', '.doc'}
    
    SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | PDF_EXTENSIONS | WORD_EXTENSIONS
    
    def __init__(self):
        pass
    
    def read_file(self, file_path: str) -> Optional[str]:
        """
        读取文件内容，自动识别格式
        
        Args:
            file_path: 文件路径
            
        Returns:
            文本内容，失败返回 None
        """
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            return None
        
        ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if ext in self.TEXT_EXTENSIONS:
                return self._read_text(file_path)
            elif ext in self.PDF_EXTENSIONS:
                return self._read_pdf(file_path)
            elif ext in self.WORD_EXTENSIONS:
                return self._read_word(file_path)
            else:
                # 尝试作为文本读取
                print(f"未知格式 {ext}，尝试作为文本文件读取")
                return self._read_text(file_path)
        except Exception as e:
            print(f"读取文件失败 {file_path}: {e}")
            return None
    
    def _read_text(self, file_path: str) -> Optional[str]:
        """读取纯文本文件"""
        # 尝试多种编码
        encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin-1']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                return content
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"读取文本文件失败 ({encoding}): {e}")
                return None
        
        print(f"无法解码文件: {file_path}")
        return None
    
    def _read_pdf(self, file_path: str) -> Optional[str]:
        """读取 PDF 文件"""
        try:
            # 尝试使用 PyPDF2
            try:
                import PyPDF2
                return self._read_pdf_pypdf2(file_path)
            except ImportError:
                pass
            
            # 尝试使用 pdfplumber
            try:
                import pdfplumber
                return self._read_pdf_plumber(file_path)
            except ImportError:
                pass
            
            # 尝试使用 fitz (PyMuPDF)
            try:
                import fitz
                return self._read_pdf_fitz(file_path)
            except ImportError:
                pass
            
            print("PDF 读取失败：请安装以下库之一：PyPDF2, pdfplumber, PyMuPDF")
            print("推荐安装: pip install pdfplumber")
            return None
            
        except Exception as e:
            print(f"读取 PDF 文件失败: {e}")
            return None
    
    def _read_pdf_pypdf2(self, file_path: str) -> str:
        """使用 PyPDF2 读取 PDF"""
        import PyPDF2
        
        text = []
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
        
        return '\n\n'.join(text)
    
    def _read_pdf_plumber(self, file_path: str) -> str:
        """使用 pdfplumber 读取 PDF"""
        import pdfplumber
        
        text = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text.append(page_text)
        
        return '\n\n'.join(text)
    
    def _read_pdf_fitz(self, file_path: str) -> str:
        """使用 PyMuPDF (fitz) 读取 PDF"""
        import fitz
        
        doc = fitz.open(file_path)
        text = []
        for page in doc:
            page_text = page.get_text()
            if page_text:
                text.append(page_text)
        doc.close()
        
        return '\n\n'.join(text)
    
    def _read_word(self, file_path: str) -> Optional[str]:
        """读取 Word 文件"""
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.docx':
            return self._read_docx(file_path)
        elif ext == '.doc':
            return self._read_doc(file_path)
        
        return None
    
    def _read_docx(self, file_path: str) -> Optional[str]:
        """读取 .docx 文件"""
        try:
            from docx import Document
            
            doc = Document(file_path)
            paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
            return '\n\n'.join(paragraphs)
        except ImportError:
            print("读取 .docx 失败：请安装 python-docx 库")
            print("安装命令: pip install python-docx")
            return None
        except Exception as e:
            print(f"读取 .docx 文件失败: {e}")
            return None
    
    def _read_doc(self, file_path: str) -> Optional[str]:
        """读取旧版 .doc 文件"""
        try:
            # 尝试使用 textract
            import textract
            text = textract.process(file_path).decode('utf-8')
            return text
        except ImportError:
            pass
        
        try:
            # 尝试使用 antiword (需要系统安装)
            import subprocess
            result = subprocess.run(
                ['antiword', file_path],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                return result.stdout
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        
        print("读取 .doc 失败：请安装 textract 或 antiword")
        print("推荐安装: pip install textract")
        return None
    
    @classmethod
    def get_supported_formats(cls) -> str:
        """获取支持的格式列表"""
        formats = {
            '文本文件': ', '.join(sorted(cls.TEXT_EXTENSIONS)),
            'PDF 文件': ', '.join(sorted(cls.PDF_EXTENSIONS)),
            'Word 文件': ', '.join(sorted(cls.WORD_EXTENSIONS)),
        }
        
        result = "支持的文档格式：\n"
        for category, exts in formats.items():
            result += f"  - {category}: {exts}\n"
        return result


# 便捷函数
def read_document(file_path: str) -> Optional[str]:
    """
    便捷函数：读取文档内容
    
    Args:
        file_path: 文件路径
        
    Returns:
        文本内容
    """
    reader = DocumentReader()
    return reader.read_file(file_path)
