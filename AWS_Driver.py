import sys
import os
import paramiko


#file_name = "NB_Bernoulli_manyDays.py"
#file_name = "XGBoost2_gbtree_manyDays_FeatureSelection.py"
file_name = "XGBoost2_gbtree_manyDays_FeatureImportancePlot2.py"
#file_name = "Test.py"
path_to_upload = "Rodrigo"
overwrite = False
download = True
download_aws = "/home/ubuntu/Rodrigo/test_XGBDmatrx_AWS.ods"
download_local = "/home/rmendoza/Desktop/test_XGBDmatrx_AWS.ods"

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(hostname="174.129.176.95", port=22, username="ubuntu", key_filename="/home/rmendoza/Desktop/rips")

if overwrite:
    upload_name = file_name
else:
    upload_name = "tmp.py"
sftp = client.open_sftp()
sftp.put(file_name, os.path.join("/home/ubuntu", path_to_upload, upload_name))


cmd = "cd {};/home/ubuntu/anaconda2/bin/python2.7 {}".format(path_to_upload, upload_name)
stdin, stdout, stderr = client.exec_command(cmd)
while True:
    output = stdout.readline()
    if output == '':
        break
    if output:
        sys.stdout.write(output)
        sys.stdout.flush()

for line in stderr:
    sys.stdout.write(line)
    sys.stdout.flush()

if download:
    sftp.get(download_aws, download_local)

if not overwrite:
    cmd = "cd {};rm {}".format(path_to_upload, upload_name)

sftp.close()
client.close()
