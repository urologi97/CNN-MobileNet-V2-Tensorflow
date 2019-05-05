- Folder "training + testing (pc)" berisikan file file yang dibutuhkan untuk proses training dan testing yang dilakukan di device yang memiliki spesifikasi yang  cukup untuk proses training (Laptop/Komputer)
  Untuk dapat menjalankan file-file dibawah dibutuhkan:
  + Python versi 3.6
  + Tensorflow versi 1.9
  + numpy
  + opencv
  + PIL
  + matplotlib
 	
	File-file dijalankan berdasarkan urutan:
		1. xml_to_csv
			Untuk menjalankan file xml_to_csv dibutuhkan folder "test_images" dan "training_images" yang berisikan gambar yang telah dilabeli menggunakan aplikasi bernama labelImg.
		2. generate_tfrecords
			Untuk mejalankan file ini, proses sebelumnya harus sudah dilakukan dengan benar
		3. train
			Untuk menjalankan file ini dibutuhkan folder "ssd_mobilenet_v2_coco_2018_03_29" yang merupakan model yang digunakan dalam proses training (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md), file config "ssd_mobilenet_v2_coco.config" dan folder training data. Proses ini dapat berjalan apabila file sebelumnya berjalan dengan benar.
		4. export_inference_graph
			Untuk menjalankan file ini dibutuhkan file "model.ckpt-xxxx" pada folder training_data. Proses ini dapat dijalankan apabila checkpoint telah berhasil dibuat dari proses sebelumnya
		5. test_video
			Untuk menjalankan file ini dibutuhkan file "label_map.pbtxt" yang berisikan keterangan penomoran kelas dan file "frozen_inference_graph.pb" pada folder inference_graph, file ini membutuhkan video bernama "test.mp4" dan akan mengeluarkan file bernama "outpy.avi"

- Folder "testing (raspberry pi)" berisikan file pengujian yang dibuat untuk raspberry pi dan raspicam
  Untuk dapat menjalankan file-file dibawah dibutuhkan:
  + Python versi 3.6
  + Tensorflow versi 1.9
  + numpy
  + opencv
  + PIL
  + picamera

	File dijalankan berdasarkan Urutan:
		1. pifeed
			Untuk dapat menjalankan file pifeed, dibutuhkan folder "inference_graph" yang dihasilkan pada proses ke-4 pada folder "training + testing (pc)", "label_map.pbtxt", dan "ssd_mobilenet_v2_coco.config"





