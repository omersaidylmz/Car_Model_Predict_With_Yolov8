#  <b> <span style='color:#16C2D5'> </span> Yapay Zeka İle Araba Model Tahmini </b> 

![screen-21 51 06 21 10 2023](https://github.com/omersaidd/Car_Model_Predict_With_Yolov8/assets/138215648/db621066-ef47-474f-9ced-658c6ffd870c)


##  <b> <span style='color:#16C2D5'> </span> Proje Açıklaması </b>  
Bu proje, YOLO (You Only Look Once) algoritması kullanarak araba markalarını görüntüler üzerinden tahmin etmeyi amaçlamaktadır. Proje, derin öğrenme ve nesne tanıma alanındaki gelişmeleri kullanarak bir nesne tanıma modeli oluşturmayı hedefler.
 
##  <b> <span style='color:#16C2D5'> </span> Projemizin Hedefleri  </b>
* Araba markalarını görüntüler üzerinden doğru bir şekilde tanıma ve tahmin etme.
* YOLO algoritması ile nesne tanıma yeteneklerini anlama ve uygulama.
* Farklı araba markaları için özelleştirilmiş modeller oluşturma ve performanslarını değerlendirme.

##  <b> <span style='color:#16C2D5'> </span> Kullanılan Teknolojiler ve Kütüphaneler </b>
* Python
* YOLO
* Opencv
* Matplotlib
##  <b> <span style='color:#16C2D5'> </span> Overfitting ve Data Augmentation Görüntü Ön İşleme </b>

Bu proje de, araba markalarını doğru bir şekilde tahmin etmek için derin öğrenme modelleri kullanırken karşılaşılan iki önemli zorluk olan overfitting (aşırı uyum) ve data augmentation (veri artırma) konularına odaklanır.

###  <b> <span style='color:#16C2D5'> </span> Overfitting (Aşırı Uyum) Nedir? </b>
Overfitting, modelin eğitim verilerine aşırı uyum sağlayarak, eğitim verileri üzerinde yüksek performans gösterirken yeni, görülmemiş verilere (test verileri gibi) kötü bir şekilde uyarlanması durumudur. Bu projede, overfitting'i önlemek için kullanılan yöntemler üzerinde titizlikle çalışıldı. Regularizasyon teknikleri, dropout katmanları ve veri artırma gibi stratejiler, modelin aşırı uyum riskini en aza indirmek için uygulandı.

 ###  <b> <span style='color:#16C2D5'> </span> Data Augmentation (Veri Artırma) Nedir? </b>
Data augmentation, mevcut eğitim verilerini manipüle etme yöntemidir. Bu yöntem, mevcut veri setini çeşitli dönüşümler (örneğin, döndürme, yansıtma, yakınlaştırma) uygulayarak yapay veri örnekleri oluşturarak veri setini genişletmeyi amaçlar. Bu, modelin farklı varyasyonlarla eğitilmesine olanak tanır, böylece gerçek dünya verilerine daha iyi uyum sağlar. Bu projede, veri artırma teknikleri ile modelin performansı ve genelleme yeteneği artırıldı.

Bu ön işleme adımları sayesinde, aşırı uyum riski en aza indirgenmiş ve modelin daha geniş bir veri yelpazesi üzerinde doğru tahminler yapabilme yeteneği artırılmıştır.

Bu açıklamalar, projenizin önemli teknik detaylarını vurgularken kullanılan ön işleme yöntemlerini açıklamaktadır. Lütfen projenizin spesifik ihtiyaçlarına uygun olarak bu açıklamaları özelleştirebilirsiniz.

 ##  <b> <span style='color:#16C2D5'> </span> Eğitim ve Kayıp Fonksiyonu (Loss Function) </b>
Bu projede kullanılan eğitim süreci, modelin doğruluğunu ölçmek ve optimize etmek için bir kayıp fonksiyonu kullanır. Kayıp fonksiyonu, modelin tahminlerinin gerçek değerlerden ne kadar uzak olduğunu ölçen bir metriktir. Eğitim süreci boyunca, bu kayıp fonksiyonu minimize edilmeye çalışılır. Projede genellikle yaygın olarak kullanılan kayıp fonksiyonlarından biri seçilmiş olabilir, örneğin, cross-entropy loss (çapraz entropi kaybı) sıkça görülen bir tercih olabilir.

 ###  <b> <span style='color:#16C2D5'> </span> Loss Fonksiyonunun Önemi </b>
Modelin Performans Değerlendirmesi: Kayıp fonksiyonu, modelin eğitim verilerine ne kadar iyi uyduğunu ölçer. Düşük kayıp değeri, modelin eğitim verilerine daha iyi uyduğunu ve daha doğru tahminler yapabileceğini gösterir.
Aşırı Uyum Kontrolü: Kayıp fonksiyonu, modelin aşırı uyum (overfitting) yapma riskini değerlendirmede de kullanılır. Eğitim verilerine çok iyi uyan, ancak yeni verilere uygun olmayan bir modelin kayıp fonksiyonu, eğitim verilerinde düşük ancak test verilerinde yüksek olabilir.

###  <b> <span style='color:#16C2D5'> </span> Loss Değerlerinin İzlenmesi </b>
Eğitim süreci boyunca kayıp değerlerinin nasıl değiştiğini izlemek, modelin ne kadar hızlı öğrendiğini ve eğitim verilerine ne kadar uygun olduğunu anlamak için önemlidir. Bu izleme, modelin aşırı uyum yapmadığından ve doğru bir şekilde genelleme yaptığından emin olmak için gereklidir.
Eğitim sürecinde kayıp değerlerinin düzenli olarak izlenmesi, modelin performansını değerlendirmek ve gerekirse eğitim parametrelerini ayarlamak için önemli bir geri bildirim sağlar. Aşağıda Loss değerlerini kontrol ettiğimizde Train, Test ve Validasyon içn iyi değerler aldığını görebiliriz.

![screen-21 59 03 21 10 2023](https://github.com/omersaidd/Car_Model_Predict_With_Yolov8/assets/138215648/8a732510-dec0-4bec-af49-63cc842f15cb)

##  <b> <span style='color:#16C2D5'> </span> Konfüzyon Matrisi (Confusion Matrix)  </b>
Bu projede sınıflandırma modelinin performansını değerlendirmek için konfüzyon matrisi kullanılır. Konfüzyon matrisi, bir sınıflandırma algoritmasının doğruluğunu değerlendirmek için kullanılan bir metriktir. Genellikle 2 veya daha fazla sınıf içeren problemlerde kullanılır ve tahmin edilen sınıflar ile gerçek sınıflar arasındaki ilişkiyi gösterir.

Bir konfüzyon matrisi genellikle dört ana terim içerir:

True Positive (TP): Modelin doğru bir şekilde pozitif sınıfı tahmin ettiği durum sayısı.
True Negative (TN): Modelin doğru bir şekilde negatif sınıfı tahmin ettiği durum sayısı.
False Positive (FP): Modelin yanlış bir şekilde pozitif sınıfı tahmin ettiği durum sayısı.
False Negative (FN): Modelin yanlış bir şekilde negatif sınıfı tahmin ettiği durum sayısı.

##  <b> <span style='color:#16C2D5'> </span> Konfüzyon Matrisi Nasıl Yorumlanır?  </b>
* Doğruluk (Accuracy): (TP + TN) / (TP + TN + FP + FN) formülü ile hesaplanır ve modelin doğru tahmin yüzdesini verir.
* Hassasiyet (Precision): TP / (TP + FP) formülü ile hesaplanır ve pozitif olarak tahmin edilenlerin gerçekten pozitif olma yüzdesini verir.
* Duyarlılık (Recall veya Sensitivity): TP / (TP + FN) formülü ile hesaplanır ve gerçek pozitiflerin ne kadarının doğru tahmin edildiğini gösterir.
* Bu metrikler, modelin performansını değerlendirirken kullanılır ve modelin doğru tahminlerini, yanlış pozitifleri ve yanlış negatifleri anlamak için önemlidir.

![screen-21 58 15 21 10 2023](https://github.com/omersaidd/Car_Model_Predict_With_Yolov8/assets/138215648/4698736d-db81-4a59-a8c1-a2582371b3b8)


Matrisi incelediğimizde tahminlerin doğruluğu ve sınıfların doğru tahmin edildiği gözükmektedir.
##  <b> <span style='color:#16C2D5'> </span> Sonuçlar  </b>
Bu proje, farklı araba modellerinin doğru bir şekilde sınıflandırmak için derin öğrenme tekniklerini kullanma amacıyla gerçekleştirildi. Yapılan çalışmalar sonucunda şu önemli sonuçlar elde edildi:

* Yüksek Doğruluk Oranı: Geliştirilen model, test verilerinde yüksek doğruluk oranları elde etti. Bu, modelin verilen MR görüntülerindeki tümörleri doğru bir şekilde sınıflandırma yeteneğini göstermektedir.

* Aşırı Uyumun Önlenmesi: Overfitting'i önlemek için uygun önlemler alındı. Regularizasyon teknikleri ve dropout katmanları kullanılarak modelin aşırı uyum riski minimize edildi.

* Data Augmentation'un Etkisi: Veri artırma teknikleri, sınırlı veri seti ile çalışırken modelin performansını artırmak için etkili bir şekilde kullanıldı. Bu, modelin daha geniş bir veri yelpazesi üzerinde doğru tahminler yapabilme yeteneğini geliştirdi.

* Confusion Matrix Analizi: Modelin performansı, confusion matrix analizi ile detaylı bir şekilde değerlendirildi. Bu analiz, modelin pozitif ve negatif sınıfları ne kadar doğru tahmin ettiğini açıkça gösterdi ve modelin güvenilirliğini belirledi.

![screen-22 01 28 21 10 2023](https://github.com/omersaidd/Car_Model_Predict_With_Yolov8/assets/138215648/1ea56a35-5d49-4464-b783-57330358968c)


Bu sonuçlar, projenin iyi bir doğruluk oranı ve yakaladığını ve doğru bir şekilde sınıflandırma yeteneğine sahip bir modelin başarıyla geliştirildiğini göstermektedir. Elde edilen bu başarılar, gelecekteki havacılık uygulamalarında geliştirilerek kullanılabilir ve otomotiv sektörüne önemli bir katkı sağlayabilir.
