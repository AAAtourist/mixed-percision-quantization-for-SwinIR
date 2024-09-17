
python basicsr/test.py -opt options/rep_quant_SwinIR_light_x3.yml --force_yml name=test_repq_x3_4bit bit=4

python basicsr/test.py -opt options/rep_quant_SwinIR_light_x4.yml --force_yml name=test_repq_x4_2bit bit=2
python basicsr/test.py -opt options/rep_quant_SwinIR_light_x4.yml --force_yml name=test_repq_x4_3bit bit=3
python basicsr/test.py -opt options/rep_quant_SwinIR_light_x4.yml --force_yml name=test_repq_x4_4bit bit=4