vai_q_tensorflow quantize --input_frozen_graph path/to/yolov4-tiny-512.pb \
			  --input_fn utils.calib_input \
			  --output_dir ./quant \
	          --input_nodes image_input \
			  --output_nodes conv2d_17/BiasAdd,conv2d_20/BiasAdd\
			  --input_shapes ?,512,512,3 \
			  --gpu 0 \
			  --calib_iter 1000 \
