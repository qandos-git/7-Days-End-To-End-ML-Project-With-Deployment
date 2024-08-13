import sys

def error_message_details(error, error_detail:sys):
    '''Extracts detailed information about an error, including the filename,
    line number, and error message.

    Args:
        error (Exception): The caught exception.
        error_detail (module): The sys module to extract traceback information.
        #Note: we use sys module to access the traceback object that contains infrmation about the error

    Returns:
        str: A formatted error message with details about the error location and message.'''
   
    # Get the traceback object from the error details sys module
    _, _, exc_tb = error_detail.exc_info()
    
    # Extract the filename and line number from the traceback object
    file_name = exc_tb.tb_frame.f_code.co_filename
    
    # Create a formatted error message
    error_message = "Error occured in Python script name[{0}], line number [{1}] error message [{2}]".format(
        file_name,
        exc_tb.tb_lineno,
        str(error)
        )

    return error_message

class CustomException(Exception):
    """A custom exception class that captures detailed error information."""
    def __init__(self, error_message, error_detail:sys):
        """Initializes the CustomException with a detailed error message.

        Args:
            error_message (module): The Exception module, a message describing the error.
            error_detail (module): The sys module to extract traceback information.
        """
        super().__init__(error_message)
        self.error_message = error_message_details(error_message, error_detail=error_detail)

    def __str__(self):
        """Returns the detailed error message when the exception is printed."""
        return self.error_message
    

