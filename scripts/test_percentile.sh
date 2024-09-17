python -u basicsr/test.py -opt options/test_quant_SwinIR_light_x2.yml --force_yml name=test_percentile_x2_4bit quantization:input_quant_params:n_bits=4 quantization:weight_quant_params:n_bits=4
python -u basicsr/test.py -opt options/test_quant_SwinIR_light_x2.yml --force_yml name=test_percentile_x2_3bit quantization:input_quant_params:n_bits=3 quantization:weight_quant_params:n_bits=3
python -u basicsr/test.py -opt options/test_quant_SwinIR_light_x2.yml --force_yml name=test_percentile_x2_2bit quantization:input_quant_params:n_bits=2 quantization:weight_quant_params:n_bits=2

python -u basicsr/test.py -opt options/test_quant_SwinIR_light_x3.yml --force_yml name=test_percentile_x3_4bit quantization:input_quant_params:n_bits=4 quantization:weight_quant_params:n_bits=4
python -u basicsr/test.py -opt options/test_quant_SwinIR_light_x3.yml --force_yml name=test_percentile_x3_3bit quantization:input_quant_params:n_bits=3 quantization:weight_quant_params:n_bits=3
python -u basicsr/test.py -opt options/test_quant_SwinIR_light_x3.yml --force_yml name=test_percentile_x3_2bit quantization:input_quant_params:n_bits=2 quantization:weight_quant_params:n_bits=2

python -u basicsr/test.py -opt options/test_quant_SwinIR_light_x4.yml --force_yml name=test_percentile_x4_4bit quantization:input_quant_params:n_bits=4 quantization:weight_quant_params:n_bits=4
python -u basicsr/test.py -opt options/test_quant_SwinIR_light_x4.yml --force_yml name=test_percentile_x4_3bit quantization:input_quant_params:n_bits=3 quantization:weight_quant_params:n_bits=3
python -u basicsr/test.py -opt options/test_quant_SwinIR_light_x4.yml --force_yml name=test_percentile_x4_2bit quantization:input_quant_params:n_bits=2 quantization:weight_quant_params:n_bits=2