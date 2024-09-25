from enum import Enum, unique


@unique
class ProgramLanguage(Enum):
    C = 'c'
    Cpp = 'cpp'
    CSharp = 'cs'
    Java = 'java'
    Python = 'py'
    JavaScript = 'js'
    Ruby = 'rb'
    Rust = 'rs'
    Others = 'other'

    @staticmethod
    def match(file_name):
        # 获取文件名的后缀名
        file_extension = file_name.split('.')[-1].lower()
        
        # 根据后缀名匹配编程语言
        if file_extension == ProgramLanguage.C.value:
            return ProgramLanguage.C
        elif file_extension == ProgramLanguage.Cpp.value:
            return ProgramLanguage.Cpp
        elif file_extension == ProgramLanguage.CSharp.value:
            return ProgramLanguage.CSharp
        elif file_extension == ProgramLanguage.Java.value:
            return ProgramLanguage.Java
        elif file_extension == ProgramLanguage.Python.value:
            return ProgramLanguage.Python
        elif file_extension == ProgramLanguage.JavaScript.value:
            return ProgramLanguage.JavaScript
        elif file_extension == ProgramLanguage.Ruby.value:
            return ProgramLanguage.Ruby
        elif file_extension == ProgramLanguage.Rust.value:
            return ProgramLanguage.Rust
        else:
            return ProgramLanguage.Others
        
        
