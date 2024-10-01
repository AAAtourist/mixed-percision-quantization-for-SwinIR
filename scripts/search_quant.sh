python -u basicsr/test.py -opt options/test_quant_SwinIR_light_x2.yml --force_yml name=search_quant_x2_4bit quantization:bits_candidate=[4]
python -u basicsr/test.py -opt options/test_quant_SwinIR_light_x2.yml --force_yml name=search_quant_x2_3bit quantization:bits_candidate=[3]
python -u basicsr/test.py -opt options/test_quant_SwinIR_light_x2.yml --force_yml name=search_quant_x2_2bit quantization:bits_candidate=[2]
python -u basicsr/test.py -opt options/test_quant_SwinIR_light_x2.yml --force_yml name=search_quant_x2_8bit quantization:bits_candidate=[8]

python -u basicsr/test.py -opt options/test_quant_SwinIR_light_x3.yml --force_yml name=search_quant_x3_4bit quantization:bits_candidate=[4]
python -u basicsr/test.py -opt options/test_quant_SwinIR_light_x3.yml --force_yml name=search_quant_x3_3bit quantization:bits_candidate=[3]
python -u basicsr/test.py -opt options/test_quant_SwinIR_light_x3.yml --force_yml name=search_quant_x3_2bit quantization:bits_candidate=[2]
python -u basicsr/test.py -opt options/test_quant_SwinIR_light_x3.yml --force_yml name=search_quant_x3_8bit quantization:bits_candidate=[8]

python -u basicsr/test.py -opt options/test_quant_SwinIR_light_x4.yml --force_yml name=search_quant_x4_4bit quantization:bits_candidate=[4]
python -u basicsr/test.py -opt options/test_quant_SwinIR_light_x4.yml --force_yml name=search_quant_x4_3bit quantization:bits_candidate=[3]
python -u basicsr/test.py -opt options/test_quant_SwinIR_light_x4.yml --force_yml name=search_quant_x4_2bit quantization:bits_candidate=[2]
python -u basicsr/test.py -opt options/test_quant_SwinIR_light_x4.yml --force_yml name=search_quant_x4_8bit quantization:bits_candidate=[8]