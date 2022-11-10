import tensorflow as tf

# Load the SavedModel.
saved_model_dir = "/home/ai-team/Object_detection_models/data/fruits/saved_model/"
saved_model_obj = tf.saved_model.load(export_dir=saved_model_dir)
print(saved_model_obj.signatures.keys())
# Load the specific concrete function from the SavedModel.
concrete_func = saved_model_obj.signatures['serving_default']
# Set the shape of the input in the concrete function.
# concrete_func.inputs[0].set_shape([])
# Convert the model to a TFLite model.
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.experimental_new_converter = True
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()
open("./fruit_object.tflite", "wb").write(tflite_model)