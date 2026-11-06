# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 11/6/2026

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không 

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

1. Chép record hạng 1: `class_id`= 468 ; `class_name`= "cab" ; `rank`=1 ; `score`= 0.510915 ; `taxonomy_name`= "ImageNet-1K"

2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
Record này nói về cả bức ảnh vì không có tọa độ box trong ảnh. Nếu nó mô tả một vật thể riêng thì phải có box hoặc mask để biết “cab” nằm ở đâu.

3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
Class list đi kèm taxonomy/checkpoints, model chỉ xuất ra con số. Lớp cuối của mạng cho ra một vector điểm, với mỗi vị trí là một số. Mạng không biết “468” nghĩa là gì, nó chỉ biết “vị trí thứ 468 có điểm cao”. Taxonomy là bảng tra do con người định nghĩa trước. Với ImageNet-1K, bảng này gồm 1000 lớp, mỗi lớp có một class_id và một tên. Chỉ số 468 ứng với “cab” là do bộ dữ liệu quy ước, không phải do model học ra.

4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
class_id là con số do mô hình xuất ra, dùng để đảm bảo sự ổn định trong quá trình huấn luyện và đánh giá
class_name để người đọc hiểu ngay vật thể đó là gì, giúp duyệt nhanh và phát hiện bất thường
taxonomy_name có thể được hiểu là “đọc theo bảng tra nào?”. Cùng một chỉ số có thể mang tên khác nhau ở các bộ nhãn khác nhau (ví dụ 468 trong ImageNet-1K khác với 468 trong COCO hay một taxonomy nội bộ). Thiếu trường này thì class_id chỉ là một con số vô nghĩa.
Ba trường cùng nhau cho phép kiểm tra chéo. Nếu class_id = 468 nhưng class_name không khớp với tên của 468 trong taxonomy_name đã khai báo, reviewer sẽ biết ngay có lỗi: dùng nhầm bảng tra, lệch thứ tự lớp, hoặc sửa tay sai.

5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
Gán một lớp cho vật chiếm diện tích lớn nhất trong ảnh. Nếu hai vật chênh nhau dưới 20% diện tích thì chọn vật gần tâm ảnh hơn. Nếu vẫn không phân định được, hoặc không có lớp phù hợp trong taxonomy, đánh dấu “mơ hồ” và escalate.

6. Vì sao không được lấy `score` của model làm ground truth?
Vì score là mức tự tin của model, không phải bằng chứng rằng nhãn đó đúng. Ground truth phải là thứ độc lập với model, do con người xác nhận theo guideline. Score vẫn có ích, nhưng vai trò chính của nó là để ưu tiên việc review, chứ không dùng để quyết định đúng hay sai.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`= "person" ; `score`= 0.912625 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`= 113.58 ; `bbox_height`= 279.68

2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
Góc bên trái: (385.33, 69.24), vật cách mép trái ảnh 385.33 px và cách mép trên 69.24px.
Góc dưới phải: (498.92, 348.92), vật cách mép trái ảnh 498.92px và cách mép trên 348.92px.
Chiều rộng box: 498.92 - 385.33 = 113.59 (khớp với record)
Chiều dài box: 348.92 - 69.24 = 279.68 (khớp với record)
Box này là hình chữ nhật đứng, cao gấp khoảng 2.5 lần chiều rộng.

3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
Chọn 2 ngưỡng 0.20 và 0.60, mỗi ngưỡng có các prediction như sau:
threshold=0.20: 17 vật thể → ['person', 'bowl', 'bowl', 'oven', 'oven', 'person', 'bowl', 'bowl', 'cup', 'cup', 'bowl', 'spoon', 'potted plant', 'spoon', 'dining table', 'spoon', 'bottle']
threshold=0.60: 6 vật thể → ['person', 'bowl', 'bowl', 'oven', 'oven', 'person']

4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
Khi hạ ngưỡng score, reviewer cần phải duyệt nhiều box hơn, chủ yếu là các box điểm thấp, đồng thời loại bỏ các box false positive, trùng lập, và kiểm tra độ khớp của box. Ở một số trường hợp, reviewer phải phán đoán các trường hợp mơ hồ, ví dụ như người ở bị che khuất, người nhỏ ở rất xa, hay người bị cắt một mép, và quyết định có tính là "person" hay không, theo quy ước của nhãn. Đổi lại, hạ ngưỡng giúp giảm bỏ sót (tăng recall), vì có thêm cơ hội bắt được những người mà mô hình ít tự tin. Tóm lại, đây là sự đánh đổi giữa công sức review và số lượng đối tượng bị bỏ sót.

5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
Vẽ một hình chữ nhật sát nhất có thể quanh phần vật thể nhìn thấy được, sao cho mỗi cạnh vừa chạm điểm ngoài cùng của vật thể đó, không chừa khoảng trống thừa và không cắt mất phần nào. Mỗi vật vẽ một box riêng, trong trường hợp bị che khuất hoặc bị cắt ở mép ảnh thì chỉ box phần nhìn thấy được, không đoán phần bị che.

6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
Vật bị cắt mép là "person" có score = 0.61 ở mép trái dưới ảnh
Mình sẽ escalate lên Lab Coach, chưa viết thành quy tắc. Hiện tại, box bao quanh sát phần thấy được, nhưng không nói có nên gán nhãn “person” hay không khi chỉ thấy một bàn tay. Người gán nhãn A có thể vẽ box quanh bàn tay, người B có thể bỏ qua vì không đủ nhận ra là người. Điều này sẽ khiến dữ liệu sẽ lệch nhau. Việc quyết định bao nhiêu phần cơ thể để đủ tính là "person" ảnh hưởng đến cả việc huấn luyện và đánh giá mô hình, nên cần có người có thẩm quyền chốt. 
Mình đề xuất phương hướng giải quyết là gửi Lab Coach kèm ảnh crop và câu hỏi rõ ràng: “Khi chỉ thấy tay hoặc cẳng tay thò vào từ mép ảnh, có gán nhãn person không? Nếu có, ngưỡng tối thiểu là gì?” Sau khi có câu trả lời, mới thêm một dòng vào guideline, ví dụ: “Chỉ gán person khi thấy được ít nhất các bộ phận gì (đầu, hoặc thân trên...)?”. Lúc đó ca này mới chuyển từ escalation thành quy tắc.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`= kitchen-001 ; `class_name`= "person" ; `score`= 0.899318 ; số điểm polygon=`polygon_point_count`= 348 ; 3–5 điểm đầu của `polygon_xy`= [446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]

2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
Box chỉ cho biết vật nằm trong hình chữ nhật nào. Polygon cho biết vật thực sự chiếm những pixel nào: 
-Hình dạng thật của vật, như ở record đã chọn: đầu tròn, vai, tay chống hông, hai chân tách nhau. Box người đứng chứa rất nhiều nền thừa ở các góc.
-Diện tích chính xác: tính được phần pixel thuộc vật, không phải diện tích hình chữ nhật.
-Phân tách vật chồng nhau: hai người đứng sát nhau có box chồng lên nhau, nhưng polygon tách được ranh giới.
-Dùng cho segmentation: từ polygon dựng được mask, mà box thì không.

3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
instance_id là tên định danh của một đối tượng cụ thể trong một ảnh, dùng để phân biệt vật này với vật kia cùng lớp. Ví dụ ảnh có hai “person” thì có hai instance_id khác nhau, còn class_name giống nhau.
Câu “không phải class ID, không phải tracking ID” nghĩa là:
-Không phải class ID: class ID nói vật thuộc loại nào, nhiều vật dùng chung một class ID. instance_id thì mỗi vật một giá trị riêng.
-Không phải tracking ID: tracking ID nối cùng một vật qua nhiều khung hình của video. instance_id chỉ có nghĩa trong phạm vi ảnh/bộ dữ liệu này và không hứa rằng “kitchen-001” ở ảnh này là cùng người với ảnh khác.

4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
Đặt các điểm polygon dọc theo ranh giới thật giữa vật và nền, sao cho biên đi qua pixel chuyển tiếp, không nằm trong nền cũng không cắt vào vật. Dùng đủ điểm để biên không lệch quá khoảng 1-2 px so với ranh giới thật, và không vẽ phần bị che hoặc bóng đổ vào mask.

5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
Ca chọn là mask “dining table 0.58” phủ lên các vật đặt trên bàn (bát, khay bột): pixel của bát bị tính vào bàn dù bát cũng có mask riêng. Mình chọn escalate lên Lab Coach rồi mới viết thành quy tắc, vì guideline hiện có chỉ nói biên bám ranh giới vật và nền, không nói pixel chồng giữa hai vật khác lớp thuộc về bên nào. Có hai cách đọc hợp lý cho dữ liệu khác nhau: mask loại trừ nhau (bàn bị khoét chỗ có bát) hoặc cho phép chồng (bàn liền, bát nằm đè), và lựa chọn này ảnh hưởng đến cả huấn luyện lẫn cách tính IoU nên phải do người có thẩm quyền chốt. Phương hướng giải quyết là đặt câu hỏi cho Lab Coach: “Vật nhỏ đặt trên vật lớn thì mask vật lớn có bị khoét không, và bàn bếp có tính là dining table không?”. Sau khi có đáp án, thêm một dòng vào guideline, ví dụ: “Mỗi pixel chỉ thuộc một mask; mask vật lớn loại trừ vùng của vật nhỏ nằm trên nó.”
## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một nhãn cho cả ảnh: class_id + class_name + taxonomy_name, không có tọa độ. | Ảnh traffic: hạng 1 cab, top-5 toàn là xe minibus, police_van. Ảnh có nhiều xe nhưng chỉ được chọn một nhãn. Xe buýt xanh nổi nhất mà hạng 1 lại là cab. ImageNet-1K không có lớp “giao thông”. |Chọn một lớp theo tiêu chí chủ đề chính (diện tích, tiêu điểm); không có lớp phù hợp thì đánh “other/mơ hồ” hoặc escalate.  | Nhãn có đúng taxonomy và đúng chủ đề chính không. Score 0.51 thấp nên xem kỹ trước, không coi score là đáp án. Cân nhắc multi-label nếu cần từng loại xe. |
| Phát hiện vật thể | Mỗi vật một box xyxy (pixel ảnh gốc) + class_name. | “person 0.61” ở mép trái chỉ là bàn tay thò vào (bị cắt mép), chồng lên “bowl 0.38/0.50”; ngưỡng thấp sinh box dư như “bowl 0.46”. Record person 0.912625: box cao/rộng khoảng 2.46, hợp dáng người. |Vẽ box sát phần nhìn thấy, mỗi vật một box. Ca “chỉ thấy bàn tay” chưa có quy tắc nên escalate Lab Coach. | Loại dương tính giả, xử lý box trùng, kiểm tra box lệch hoặc ôm thiếu. Duyệt lướt box điểm cao, xem kỹ vùng điểm thấp. |
| Instance segmentation | Mỗi vật một polygon/mask + class_name + instance_id | dining table 0.58” tô cả mặt bàn, đè lên bát (pixel chồng thuộc ai?); “person 0.43” ở góc trái dưới nhiều khả năng dương tính giả; “oven 0.49” biên mờ vì nền tối; mask đầu bếp dính liền ở chỗ tay chạm thân áo trắng |Vẽ biên bám ranh giới vật/nền (sai số 1-2 px), không vẽ phần bị che hay bóng. Ca pixel chồng giữa hai lớp thì escalate chờ quy tắc. |Kiểm tra biên có lệch hoặc tràn sang vật khác không, instance_id có tách đúng từng vật không, lớp có đúng không. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
Chỉ dùng ảnh và file prediction mà lab cung cấp (kitchen, traffic) cho đúng mục đích của bài lab, không đưa ra ngoài hay dùng cho việc khác, và không thêm thông tin nhận dạng người nào ngoài những gì nhìn thấy trong ảnh.

2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
Mình sẽ dừng lại, không gán nhãn hay xử lý tiếp, và báo cho Lab Coach, kèm mô tả ngắn: ảnh/file nào, vì sao thấy nó ngoài phạm vi.

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
