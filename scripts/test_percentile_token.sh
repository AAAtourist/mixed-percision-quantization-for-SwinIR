python -u basicsr/test.py -opt options/test_quant_SwinIR_light_x2.yml --force_yml name=test_percentile_token_x2_4bit quantization:input_quant_params:n_bits=4 quantization:weight_quant_params:n_bits=4
python -u basicsr/test.py -opt options/test_quant_SwinIR_light_x2.yml --force_yml name=test_percentile_token_x2_3bit quantization:input_quant_params:n_bits=3 quantization:weight_quant_params:n_bits=3
python -u basicsr/test.py -opt options/test_quant_SwinIR_light_x2.yml --force_yml name=test_percentile_token_x2_2bit quantization:input_quant_params:n_bits=2 quantization:weight_quant_params:n_bits=2