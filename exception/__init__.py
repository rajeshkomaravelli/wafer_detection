import os
import sys

class exception(Exception):
    
    def __init__(self, error_message:Exception,error_detail:sys):
        super().__init__(error_message)
        self.error_message=exception.get_detailed_error_message(error_message=error_message,
                                                                       error_detail=error_detail
                                                                        )
    #super.__init__ function is used when we want to use constructor of the parent class here exception class we are using it and creating our own representation of the exception

    @staticmethod
    def get_detailed_error_message(error_message:Exception,error_detail:sys)->str:
        """
        error_message: Exception object
        error_detail: object of sys module
        """
        _,_ ,exec_tb = error_detail.exc_info()
        #This will give the line number of error
        exception_block_line_number = exec_tb.tb_frame.f_lineno
        try_block_line_number = exec_tb.tb_lineno
        file_name = exec_tb.tb_frame.f_code.co_filename
        error_message = f"""
        Error occured in script: 
        [ {file_name} ] at 
        try block line number: [{try_block_line_number}] and exception block line number: [{exception_block_line_number}] 
        error message: [{error_message}]
        """
        return error_message

    def __str__(self):
        return self.error_message

    #this function prints a message when class object is used inside a print function
    def __repr__(self) -> str:
        return exception.__name__.str()
    #this function effect can be seen in jupyter notebook when we create a object of this class the message inside this function is printed