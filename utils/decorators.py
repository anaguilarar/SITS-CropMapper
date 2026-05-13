import os

def check_path(func):
    """
    A decorator that checks the existence of a specified path, creates it if necessary, 
    and constructs the full file path with the correct suffix.

    Parameters:
    func (function): A function that requires path, filename (fn), and suffix as arguments.

    Returns:
    function: A wrapper function that adds path validation and adjustment to the original function.

    Raises:
    ValueError: If the specified path cannot be used or created.
    """
    
    def inner(file, path, **kwargs):
   
        if not os.path.exists(path):
            raise ValueError(f"Unable to use the specified path: {path}")
                  
        return func(file, path=path,  **kwargs)
    
    return inner

def check_output_fn(func):
    """
    A decorator that checks the existence of a specified path, creates it if necessary, 
    and constructs the full file path with the correct suffix.

    Parameters:
    func (function): A function that requires path, filename (fn), and suffix as arguments.

    Returns:
    function: A wrapper function that adds path validation and adjustment to the original function.

    Raises:
    ValueError: If the specified path cannot be used or created.
    """
    
    def inner(file, path, fn = None, suffix = None):
        
        if fn is None:
            fn = os.path.basename(path)
            path = os.path.dirname(path)
        try:
            if not os.path.exists(path):
                os.makedirs(path)
        except Exception as e:
            raise ValueError(f"Unable to use or create the specified path: {path}. Error: {e}")
            
        if suffix:
            fn = os.path.join(path, fn if fn.endswith(suffix) else fn + suffix)
          
        return func(file, path=path, fn=fn)
    
    return inner