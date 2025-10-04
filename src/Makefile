sq_pref_test_u280:
	cp -r ~/HMT_System_Tapa/SpinQuant_HLS_NR/parameters ./
	cp ~/HMT_System_Tapa/SpinQuant_HLS_NR/*.cpp ~/HMT_System_Tapa/SpinQuant_HLS_NR/*.h ~/HMT_System_Tapa/SpinQuant_HLS_NR/*.txt ~/HMT_System_Tapa/SpinQuant_HLS_NR/*.ini ./
	tapa g++ -- SpinQuant_Prefilling_test_tb.cpp -o SpinQuant_Prefilling_test
	./SpinQuant_Prefilling_test
	tapa compile -j 128 --top SpinQuant_Prefilling --platform xilinx_u280_gen3x16_xdma_1_202211_1 --clock-period 3.33 -f SpinQuant_Prefilling_test.h -o SpinQuant_Prefilling.xo

	mkdir RapidStream
	cp /home/jameszhang23/HMT_System_Tapa/SpinQuant_HLS_NR/RapidStream_pref_u280/*.json RapidStream/

	rapidstream-tapaopt \
	-j 32 \
	--work-dir ./RapidStream/build \
	--tapa-xo-path SpinQuant_Prefilling.xo \
	--device-config ./RapidStream/u280_device.json \
	--floorplan-config ./RapidStream/floorplan_config_test.json \
	--pipeline-config ./RapidStream/pipeline_config.json \
	--run-impl \
	--implementation-config ./RapidStream/impl_config.json \
	--connectivity-ini pref_link_config_u280_test.ini

	./SpinQuant_Prefilling --bitstream SpinQuant_Prefilling.xo -xosim_work_dir ./work.out -xosim_save_waveform


sq_pref_u280:
	cp -r ~/HMT_System_Tapa/SpinQuant_HLS_NR/parameters ./
	cp ~/HMT_System_Tapa/SpinQuant_HLS_NR/*.cpp ~/HMT_System_Tapa/SpinQuant_HLS_NR/*.h ~/HMT_System_Tapa/SpinQuant_HLS_NR/*.txt ~/HMT_System_Tapa/SpinQuant_HLS_NR/*.ini ./
	tapa g++ -- SpinQuant_Prefilling_tb.cpp -o SpinQuant_Prefilling
	./SpinQuant_Prefilling
	rm
	mkdir RapidStream
	cp /home/jameszhang23/HMT_System_Tapa/SpinQuant_HLS_NR/RapidStream_pref_u280/*.json RapidStream/

	rapidstream-tapaopt \
	-j 32 \
	--work-dir ./RapidStream/build \
	--tapa-xo-path SpinQuant_Prefilling.xo \
	--device-config ./RapidStream/u280_device.json \
	--floorplan-config ./RapidStream/floorplan_config.json \
	--pipeline-config ./RapidStream/pipeline_config.json \
	--run-impl \
	--implementation-config ./RapidStream/impl_config.json \
	--connectivity-ini pref_link_config_u280.ini

	./SpinQuant_Prefilling --bitstream SpinQuant_Prefilling.xo -xosim_work_dir ./work.out -xosim_save_waveform


sq_pref_u280_mem_opt:
	cp -r ~/HMT_System_Tapa/SpinQuant_HLS_NR/parameters ./
	cp ~/HMT_System_Tapa/SpinQuant_HLS_NR/*.cpp ~/HMT_System_Tapa/SpinQuant_HLS_NR/*.h ~/HMT_System_Tapa/SpinQuant_HLS_NR/*.txt ~/HMT_System_Tapa/SpinQuant_HLS_NR/*.ini ./
	tapa g++ -- SpinQuant_Prefilling_mem_opt_tb.cpp -o SpinQuant_Prefilling_mem_opt
	./SpinQuant_Prefilling_mem_opt
	tapa compile -j 128 --top SpinQuant_Prefilling --platform xilinx_u280_gen3x16_xdma_1_202211_1 --clock-period 3.33 -f SpinQuant_Prefilling_mem_opt.h -o SpinQuant_Prefilling.xo

	mkdir RapidStream
	cp /home/jameszhang23/HMT_System_Tapa/SpinQuant_HLS_NR/RapidStream_pref_u280/*.json RapidStream/

	rapidstream-tapaopt \
    -j 32 \
    --work-dir ./RapidStream/build \
    --tapa-xo-path SpinQuant_Prefilling.xo \
    --device-config ./RapidStream/u280_device.json \
    --floorplan-config ./RapidStream/floorplan_config_mem_opt_96.json \
    --pipeline-config ./RapidStream/pipeline_config.json \
    --run-impl \
    --implementation-config ./RapidStream/impl_config.json \
    --connectivity-ini pref_link_config_u280_mem_opt_96.ini





sq_dec_u280:
	cp -r ~/HMT_System_Tapa/SpinQuant_HLS_NR/parameters ./
	cp ~/HMT_System_Tapa/SpinQuant_HLS_NR/*.cpp ~/HMT_System_Tapa/SpinQuant_HLS_NR/*.h ~/HMT_System_Tapa/SpinQuant_HLS_NR/*.txt ~/HMT_System_Tapa/SpinQuant_HLS_NR/*.ini ./
	tapa g++ -- SpinQuant_Decoding_tb.cpp -o SpinQuant_Decoding
	./SpinQuant_Decoding
	tapa compile -j 128 --top SpinQuant_Decoding --platform xilinx_u280_gen3x16_xdma_1_202211_1 --clock-period 3.33 -f SpinQuant_Decoding.h -o SpinQuant_Decoding.xo

	mkdir RapidStream
	cp /home/jameszhang23/HMT_System_Tapa/SpinQuant_HLS_NR/RapidStream_dec_u280/*.json RapidStream/

	rapidstream-tapaopt \
    -j 32 \
    --work-dir ./RapidStream/build \
    --tapa-xo-path SpinQuant_Decoding.xo \
    --device-config ./RapidStream/u280_device.json \
    --floorplan-config ./RapidStream/floorplan_config_unroll.json \
    --pipeline-config ./RapidStream/pipeline_config.json \
    --run-impl \
    --implementation-config ./RapidStream/impl_config.json \
    --connectivity-ini dec_link_config_u280_unroll.ini


	./SpinQuant_Decoding --bitstream SpinQuant_Decoding.xo -xosim_work_dir ./work.out -xosim_save_waveform


	v++ -o SpinQuant_Decoding_xilinx_u280_gen3x16_xdma_1_202211_1.hw_emu.xclbin \
	--link \
	--target hw_emu \
	--kernel SpinQuant_Decoding \
	--kernel_frequency 300\
	--config dec_link_config_u280_unroll_emu.ini \
	--platform xilinx_u280_gen3x16_xdma_1_202211_1 \
	--report_level 2 \
	--temp_dir ./temp_prj \
	--save-temps \
	SpinQuant_Decoding.xo

sq_dec_u280_mem_opt:
	cp -r ~/HMT_System_Tapa/SpinQuant_HLS_NR/parameters ./
	cp ~/HMT_System_Tapa/SpinQuant_HLS_NR/*.cpp ~/HMT_System_Tapa/SpinQuant_HLS_NR/*.h ~/HMT_System_Tapa/SpinQuant_HLS_NR/*.txt ~/HMT_System_Tapa/SpinQuant_HLS_NR/*.ini ./
	tapa g++ -- SpinQuant_Decoding_mem_opt_tb.cpp -o SpinQuant_Decoding_mem_opt
	./SpinQuant_Decoding_mem_opt
	tapa compile -j 128 --top SpinQuant_Decoding --platform xilinx_u280_gen3x16_xdma_1_202211_1 --clock-period 3.33 -f SpinQuant_Decoding_mem_opt.h -o SpinQuant_Decoding.xo

	mkdir RapidStream
	cp /home/jameszhang23/HMT_System_Tapa/SpinQuant_HLS_NR/RapidStream_dec_u280/*.json RapidStream/

	rapidstream-tapaopt \
	-j 32 \
	--work-dir ./RapidStream/build \
	--tapa-xo-path SpinQuant_Decoding.xo \
	--device-config ./RapidStream/u280_device.json \
	--floorplan-config ./RapidStream/floorplan_config_mem_opt_2.json \
	--pipeline-config ./RapidStream/pipeline_config.json \
	--run-impl \
	--implementation-config ./RapidStream/impl_config.json \
	--connectivity-ini dec_link_config_u280_mem_opt.ini

	./SpinQuant_Decoding_mem_opt --bitstream SpinQuant_Decoding.xo -xosim_work_dir ./work.out -xosim_save_waveform


	v++ -o SpinQuant_Decoding_mem_opt_xilinx_u280_gen3x16_xdma_1_202211_1.hw_emu.xclbin \
	--link \
	--target hw_emu \
	--kernel SpinQuant_Decoding \
	--kernel_frequency 300\
	--config dec_link_config_u280_mem_opt_emu.ini \
	--platform xilinx_u280_gen3x16_xdma_1_202211_1 \
	--report_level 2 \
	--temp_dir ./temp_prj \
	--save-temps \
	SpinQuant_Decoding.xo
