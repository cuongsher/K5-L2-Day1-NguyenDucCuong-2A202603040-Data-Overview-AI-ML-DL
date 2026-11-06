# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=… ; `class_name`=… ; `rank`=1 ; `score`=… ; `taxonomy_name`=…
    {
    "sample_id": "traffic",
    "coco_image_id": 210273,
    "image_width": 640,
    "image_height": 428,
    "task": "image_classification",
    "taxonomy_name": "ImageNet-1K",
    "model_file": "yolo11n-cls.pt",
    "model_sha256": "c62d41bf9625777760018bf914d2e6cd472420ccd01706d97a61cb6c82502bd7",
    "ultralytics_version": "8.4.145",
    "rank": 1,
    "class_id": 468,
    "class_name": "cab",
    "score": 0.510915
  },
  
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
Về cả bức ảnh. Lý do:
 + Trường task là image_classification, tức là model gán một nhãn cho toàn ảnh.
 + Record không có bounding box, mask hay tọa độ nào, chỉ có image_width và image_height của cả ảnh. Nếu nói về một vật thể riêng thì phải có vị trí của vật đó.
 + Trong visuals/classification_top5.png, không có khung hay vùng tô nào đánh dấu một xe cụ thể, chỉ có một thanh điểm cho mỗi lớp. Ảnh traffic lại chứa rất nhiều vật cùng lúc (xe buýt, xe con, người đi bộ), nên nhãn “cab” nghĩa là “toàn cảnh này giống lớp cab nhất theo model”, không phải “chiếc xe này là taxi”.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
   Đi kèm taxonomy/checkpoint, không phải model tự nghĩ ra. File yolo11n-cls.pt được huấn luyện trên ImageNet-1K, nên đầu ra của nó là 1000 chỉ số cố định, mỗi chỉ số ứng với một tên lớp do taxonomy định nghĩa. Model chỉ chọn trong tập lớp đó, nên nó không thể trả ra nhãn nằm ngoài danh sách (ví dụ “taxi Hà Nội”). Đó cũng là lý do ảnh kitchen bị gán là gong hay lumbermill: không có lớp nào đúng hơn trong tập đóng này.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
   + class_id chỉ là số thứ tự, chỉ có nghĩa trong một taxonomy cụ thể. Id 468 ở ImageNet-1K là “cab”, nhưng ở bộ nhãn khác (COCO, Open Images...) cùng số đó có thể là lớp khác.
   + class_name giúp người đọc hiểu ngay, nhưng tên có thể trùng, viết khác nhau, hoặc đổi giữa các phiên bản.
   + taxonomy_name xác định “từ điển” mà hai trường kia thuộc về, nên ba trường cùng nhau mới tạo thành định danh không mơ hồ và có thể tái lập, đối chiếu, ánh xạ sang taxonomy khác sau này.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
   + Tiêu chí chọn: vật chiếm diện tích lớn nhất, vật ở trung tâm, hay chủ đề tổng thể của cảnh.
   + Thứ tự ưu tiên khi các tiêu chí mâu thuẫn, và cách xử lý khi nhiều vật ngang nhau (quy tắc phụ, hoặc đánh dấu “mơ hồ” thay vì đoán).
   + Mức chi tiết của nhãn (ví dụ “cab” hay chỉ “car”) và cách xử lý vật nhỏ hoặc bị che khuất.
   + Có cho phép nhãn “không rõ / không thuộc lớp nào” hay không, vì tập lớp của ImageNet-1K là tập đóng và không có lớp “đường phố đông xe”.
6. Vì sao không được lấy `score` của model làm ground truth?
   + score là độ tin cậy của model, không phải sự thật. Đó là xác suất softmax của một model nhỏ (yolo11n-cls), có thể sai mà vẫn tự tin. Chính biểu đồ cũng ghi “không phải chất lượng ground truth”.
   + Ở ảnh traffic, score cao nhất chỉ 0.51 (cab), còn minibus 0.16, tức model khá phân vân. Ở ảnh kitchen, model tự tin 0.42 vào gong, rõ ràng sai.
   + Nếu lấy đầu ra của model làm ground truth rồi lại đánh giá chính model đó, ta đang chấm bài bằng đáp án do chính nó viết (vòng lặp tự xác nhận), mọi lỗi và thiên lệch bị biến thành “đúng”.
   + Ground truth phải do người gán nhãn tạo ra theo guideline, có kiểm tra chéo (review hoặc đồng thuận giữa nhiều người). Kết quả model chỉ nên dùng làm gợi ý (pre-label) để người xem xét và sửa.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=… ; `score`=… ; `bbox_xyxy`=[…, …, …, …] ; `bbox_width`=… ; `bbox_height`=…
   {
    "sample_id": "kitchen",
    "coco_image_id": 397133,
    "image_width": 640,
    "image_height": 427,
    "task": "object_detection",
    "taxonomy_name": "COCO-80",
    "model_file": "yolo11n.pt",
    "model_sha256": "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1",
    "ultralytics_version": "8.4.145",
    "score_threshold": 0.35,
    "class_id": 0,
    "class_name": "person",
    "score": 0.912625,
    "coordinate_unit": "pixel",
    "bbox_format": "xyxy",
    "bbox_xyxy": [
      385.33,
      69.24,
      498.92,
      348.92
    ],
    "bbox_width": 113.58,
    "bbox_height": 279.68
  },
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
   + Góc trên-trái nằm cách mép trái ảnh khoảng 385 px và cách mép trên khoảng 69 px.
   + Góc dưới-phải nằm cách mép trái khoảng 499 px và cách mép trên khoảng 349 px.
   + Hộp rộng khoảng 114 px và cao khoảng 280 px, tức cao hơn rộng gần 2,5 lần.
   + Gốc tọa độ (0, 0) ở góc trên-trái của ảnh, x tăng sang phải, y tăng xuống dưới.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
  + Ngưỡng 0.35: 11 prediction (2 person, 5 bowl, 2 oven, 2 cup).
  + Ngưỡng 0.60: 6 prediction (2 person, 2 bowl, 2 oven). Các hộp bị loại là bowl 0.50, bowl 0.46, cup 0.45, cup 0.38 và bowl 0.38.
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
  + Kiểm tra thêm nhiều hộp hơn, nên tốn thêm thời gian cho mỗi ảnh.
  + Với mỗi hộp mới, xác nhận vật đó có thật không (loại hộp nhận nhầm, tức false positive), nhãn lớp có đúng không (ví dụ cup hay bowl), và box có ôm sát vật không.
  + Loại hộp trùng nhau hoặc chồng lên nhau, như các bowl và cup nằm sát nhau ở góc trái dưới của ảnh.
  + Tìm lỗi hay gặp ở các hộp điểm thấp: hộp quá nhỏ, quá lỏng, hoặc nằm trên vật bị che.
  + Bù lại, hạ ngưỡng giúp ít bỏ sót vật hơn (ít false negative), nhưng đổi lại phải rà soát nhiều hơn.

5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
   Mỗi vật thuộc lớp cần gán nhãn được vẽ một box riêng, ôm sát toàn bộ phần nhìn thấy của vật đó, không chừa nền thừa và không cắt mất phần nào của vật
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
   + Vật bị cắt mép hoặc bị che là tình huống lặp lại ở rất nhiều ảnh (người ở rìa khung hình, vật sau vật khác), nên cần một cách xử lý thống nhất cho mọi người gán nhãn.
   + Quy tắc có thể trả lời rõ ràng. Ví dụ: “Với vật bị cắt mép hoặc bị che, chỉ vẽ box quanh phần nhìn thấy, không đoán phần bị khuất, và gắn cờ ‘bị cắt/bị che’ nếu có thuộc tính đó. Nếu phần nhìn thấy nhỏ hơn một ngưỡng quy định thì bỏ qua.”
   + Mỗi người tự quyết một kiểu sẽ làm dữ liệu không nhất quán, trong khi guideline đủ thông tin để quyết định.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=… ; `class_name`=… ; `score`=… ; số điểm polygon=`polygon_point_count`=… ; 3–5 điểm đầu của `polygon_xy`=…
    {
    "sample_id": "kitchen",
    "coco_image_id": 397133,
    "image_width": 640,
    "image_height": 427,
    "task": "instance_segmentation",
    "taxonomy_name": "COCO-80",
    "model_file": "yolo11n-seg.pt",
    "model_sha256": "55ed65c56c91713d23e8402371c6c49a6fd84f257f7dce452e8d70e41dcbe152",
    "ultralytics_version": "8.4.145",
    "score_threshold": 0.35,
    "instance_id": "kitchen-001",
    "class_id": 0,
    "class_name": "person",
    "score": 0.899318,
    "coordinate_unit": "pixel",
    "bbox_format": "xyxy",
    "bbox_xyxy": [
      385.45,
      66.44,
      498.02,
      348.58
    ],
    "polygon_point_count": 348,
    "polygon_xy": [
      [
        446.0,
        70.0
      ],
      [
        445.0,
        71.0
      ],
      [
        444.0,
        71.0
      ],
      [
        443.0,
        72.0
      ],
      [
        442.0,
        72.0
      ],]}
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
 +  Hình dạng: cái đầu tròn, vai, hai tay chống hông, tạp dề và hai chân.
 + Phần nào trong box là vật và phần nào là nền. Ở góc box của người đầu bếp có nhiều nền (nồi, tường), còn mask thì không lấy những phần này.
 + Các khoảng hở, ví dụ khoảng trống giữa hai chân và giữa tay với thân.
 + Diện tích thật của vật (đếm số pixel trong mask). Diện tích box thì luôn lớn hơn diện tích vật.
  
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
   instance_id (ví dụ kitchen-001) là nhãn gọi tên từng vật cụ thể trong một ảnh, để phân biệt vật này với vật khác và để nhắc đến đúng một mask khi review hoặc sửa
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
   Mỗi vật được vẽ một mask riêng, đi sát theo mép ngoài của phần nhìn thấy của vật đó: không lấn ra nền, không cắt vào trong vật. Khi vật bị vật khác che, chỉ vẽ phần nhìn thấy và không đoán phần bị khuất
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
   + dùng guideline cho phần lớn tình huống này, vì vật chạm nhau, chồng lên nhau hoặc bị cắt mép gặp ở rất nhiều ảnh, nên cần một cách làm thống nhất cho mọi người gán nhãn. Ví dụ: pixel nào thuộc vật nằm phía trước thì giao cho vật đó, các mask không chồng lên nhau, vật bị cắt mép chỉ vẽ phần nhìn thấy
   + Nếu mỗi người tự quyết sẽ cho ra dữ liệu không nhất quán, nên cần Lab Coach quyết định một lần rồi cập nhật guideline cho các ca sau.


## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn cho cả ảnh: class_id + class_name + taxonomy_name. Không có box hay mask. | `traffic`: top-1 là cab (0.51), minibus 0.16, model phân vân. `kitchen` bị gán gong vì taxonomy là tập đóng. | Chọn một lớp theo guideline; không lớp nào hợp thì chọn "không rõ". | Nhãn đúng guideline và nằm trong taxonomy; không lấy score làm đáp án. |
| Phát hiện vật thể | Mỗi vật một box `bbox_xyxy` (pixel) kèm lớp. | `kitchen`: person 0.61 sát mép trái, chỉ thấy tay. Bowl/cup chồng nhau. Ngưỡng 0.35 giữ 11 box, 0.60 còn 6. | Vẽ box sát phần nhìn thấy, gán lớp, xóa box thừa, thêm box thiếu. | Box sót, thừa, trùng; box có sát vật; lớp đúng không. |
| Instance segmentation | Mỗi vật một polygon `polygon_xy` kèm lớp và `instance_id`. | `kitchen`: mask dining table chồng lên oven; potted plant 0.63 có thể sai lớp. Polygon spoon có điểm nhảy lung tung. | Vẽ biên sát phần nhìn thấy, không chồng mask; ca không rõ lớp thì escalate Lab Coach. | Biên sát, polygon không tự cắt, mask không chồng, lớp và `instance_id` đúng. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
   chỉ dùng đúng các ảnh mẫu và file đầu ra được giao cho lab, không đưa ảnh hay dữ liệu ra ngoài môi trường lab và không thêm dữ liệu cá nhân nào.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
   Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, mình dừng ngay, không mở rộng hay chia sẻ dữ liệu đó, rồi báo cho Lab Coach để họ quyết định cách xử lý.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [ ] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [ ] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
