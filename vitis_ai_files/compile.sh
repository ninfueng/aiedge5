vai_c_tensorflow --frozen_pb path/to/quantize_eval_model.pb \
                 --arch arch.json \
		         --output_dir ./compile/ \
		         --net_name yolov4_tiny_512 \
		         --options "{'mode':'normal','save_kernel':'', 'input_shape':'1,512,512,3'}"
