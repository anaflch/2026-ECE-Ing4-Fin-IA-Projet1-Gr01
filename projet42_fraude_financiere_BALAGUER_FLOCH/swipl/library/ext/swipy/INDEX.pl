/*  Creator: make/0

    Purpose: Provide index for autoload
*/

index(py_version, janus, janus).
index(py_call(?), janus, janus).
index(py_call(?,?), janus, janus).
index(py_call(?,?,?), janus, janus).
index(py_iter(?,?), janus, janus).
index(py_iter(?,?,?), janus, janus).
index(py_setattr(?,?,?), janus, janus).
index(py_free(?), janus, janus).
index(py_is_object(?), janus, janus).
index(py_is_dict(?), janus, janus).
index(py_with_gil(0), janus, janus).
index(py_gil_owner(?), janus, janus).
index(py_func(?,?,?), janus, janus).
index(py_func(?,?,?,?), janus, janus).
index(py_dot(?,?,?), janus, janus).
index(py_dot(?,?,?,?), janus, janus).
index(values(?,?,?), janus, janus).
index(keys(?,?), janus, janus).
index(key(?,?), janus, janus).
index(items(?,?), janus, janus).
index(py_shell, janus, janus).
index(py_pp(?), janus, janus).
index(py_pp(?,?), janus, janus).
index(py_pp(?,?,?), janus, janus).
index(py_object_dir(?,?), janus, janus).
index(py_object_dict(?,?), janus, janus).
index(py_obj_dir(?,?), janus, janus).
index(py_obj_dict(?,?), janus, janus).
index(py_type(?,?), janus, janus).
index(py_isinstance(?,?), janus, janus).
index(py_module_exists(?), janus, janus).
index(py_hasattr(?,?), janus, janus).
index(py_import(?,?), janus, janus).
index(py_module(?,?), janus, janus).
index(py_initialize(?,?,?), janus, janus).
index(py_lib_dirs(?), janus, janus).
index(py_add_lib_dir(?), janus, janus).
index(py_add_lib_dir(?,?), janus, janus).
index(:(op,op(200,fy,@)), janus, janus).
index(:(op,op(50,fx,#)), janus, janus).
