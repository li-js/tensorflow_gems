import tensorflow as tf

# TF initialize only uninitialized_variables
uninitialized_var_names = sess.run(tf.report_uninitialized_variables())
sess.run(tf.variables_initializer([v for v in tf.global_variables() if v.name.split(':')[0] in uninitialized_var_names]))
