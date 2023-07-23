# Virtual Environment  ve Paket Yönetimi


# Sanal ortamların listelenmesi:
# conda env list

# Sanal ortam oluşturma:
# conda create –n muyux

# Sanal ortamı aktif etme:
# conda activate muyux

# Yüklü paketlerin listelenmesi:
# conda list

# Paket yükleme:
# conda install numpy

# Aynı anda birden fazla paket yükleme:
# conda install numpy scipy pandas

# Paket silme:
# conda remove package_name

# Belirli bir versiyona göre paket yükleme:
# conda install numpy=1.20.1

# Paket yükseltme:
# conda upgrade numpy

# Tüm paketlerin yükseltilmesi:
# conda upgrade –all

# pip: pypi (python package index) paket yönetim aracı

# Paket yükleme:
# pip install pandas

# Paket yükleme versiyona göre:
# pip install pandas==1.2.1

#Sanal ortamın deactivate edilmesi ve silinmesi
#conda deactivate
#conda env remove -n MuyuX

#Sanal ortamın dışarı aktarılması ve içeri çekilmesi
#conda env export > envitoment.yaml
#conda env create -f envitoment.yaml