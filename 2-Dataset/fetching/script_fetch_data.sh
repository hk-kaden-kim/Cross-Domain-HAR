#!bin/bash

OUTPUT_ROOT="$SCRATCH/mp-dataset"

echo "Which Dataset would you like to download?"
echo "0. capture24"
echo "1. adl"
echo "2. realword"
echo "3. wisdm"
echo "4. pamap2"
echo "5. opportunity"
echo "6. forh-trace"
echo "7. selfback"
echo "8. gotov"
echo "9. harvardleo"
echo "10. householdhu"
echo "11. mendeleydaily"
echo "12. WristPPG"
echo "13. NewCastle"
echo "14. Commuting"
echo "15. Ichi14"
echo "16. Paal"
echo "17. *UNSEENDATASET"

read -p "Enter your choice : " choice

case $choice in 
    0)
        echo "Downloading capture24 dataset..."
        curl https://ora.ox.ac.uk/objects/uuid:99d7c092-d865-4a19-b096-cc16440cd001/download_file\?file_format\=\&safe_filename\=capture24.zip\&type_of_work\=Dataset --output "$OUTPUT_ROOT"/capture24.zip
        echo "Unzip capture24 dataset..."
        unzip -o -q "$OUTPUT_ROOT"/capture24.zip -d "$OUTPUT_ROOT/capture24"
        echo "Delete capture24 zip file..."
        rm "$OUTPUT_ROOT"/capture24.zip
        ;;
    1)
        echo "Downloading adl dataset..."
        curl https://archive.ics.uci.edu/ml/machine-learning-databases/00283/ADL_Dataset.zip --output "$OUTPUT_ROOT"/adl.zip
        echo "Unzip adl dataset..."
        unzip -o -q "$OUTPUT_ROOT"/adl.zip -d "$OUTPUT_ROOT/adl"
        echo "Delete adl zip file..."
        rm "$OUTPUT_ROOT"/adl.zip
        ;;
    2)
        echo "Downloading realword dataset..."
        curl http://wifo5-14.informatik.uni-mannheim.de/sensor/dataset/realworld2016/realworld2016_dataset.zip --output "$OUTPUT_ROOT"/realworld.zip
        echo "Unzip realword dataset..."
        unzip -o -q "$OUTPUT_ROOT"/realworld.zip -d "$OUTPUT_ROOT/realworld"
        echo "Delete realword zip file..."
        rm "$OUTPUT_ROOT"/realworld.zip
        ;;
    3) 
        echo "Downloading wisdm dataset..."
        curl https://archive.ics.uci.edu/static/public/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset.zip --output "$OUTPUT_ROOT"/wisdm.zip
        echo "Unzip wisdm original dataset..."
        unzip -o -q "$OUTPUT_ROOT"/wisdm.zip -d "$OUTPUT_ROOT/wisdm"
        echo "Delete wisdm zip file..."
        rm "$OUTPUT_ROOT"/wisdm.zip
        echo "Unzip wisdm-dataset..."
        unzip -o -q "$OUTPUT_ROOT"/wisdm/wisdm-dataset.zip -d "$OUTPUT_ROOT/wisdm"
        echo "Delete wisdm-dataset zip file..."
        rm "$OUTPUT_ROOT"/wisdm/wisdm-dataset.zip
        ;;
    4)
        echo "Downloading pamap2 dataset..."
        curl https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip --output "$OUTPUT_ROOT"/pamap2.zip
        echo "Unzip pamap2 original dataset..."
        unzip -o -q "$OUTPUT_ROOT"/pamap2.zip -d "$OUTPUT_ROOT/pamap2"
        echo "Delete pamap2 zip file..."
        rm "$OUTPUT_ROOT"/pamap2.zip
        echo "Unzip PAMAP2_Dataset..."
        unzip -o -q "$OUTPUT_ROOT"/pamap2/PAMAP2_Dataset.zip -d "$OUTPUT_ROOT/pamap2"
        echo "Delete PAMAP2_Dataset zip file..."
        rm "$OUTPUT_ROOT"/pamap2/PAMAP2_Dataset.zip
        ;;
    5)
        echo "Downloading opportunity dataset..."
        curl https://archive.ics.uci.edu/static/public/226/opportunity+activity+recognition.zip --output "$OUTPUT_ROOT"/opportunity.zip
        echo "Unzip opportunity dataset..."
        unzip -o -q "$OUTPUT_ROOT"/opportunity.zip -d "$OUTPUT_ROOT/opportunity"
        echo "Delete opportunity zip file..."
        rm "$OUTPUT_ROOT"/opportunity.zip
        ;;

    6)
        echo "Downloading forh-trace dataset..."
        curl https://zenodo.org/record/841301/files/FORTH_TRACE_DATASET.zip?download=1 --output "$OUTPUT_ROOT"/forth-trace.zip
        echo "Unzip forth-trace dataset..."
        unzip -o -q "$OUTPUT_ROOT"/forth-trace.zip -d "$OUTPUT_ROOT/forth-trace"
        echo "Delete forth-trace zip file..."
        rm "$OUTPUT_ROOT"/forth-trace.zip
        ;;

    7)
        if test -f ~/.kaggle/kaggle.json; then
        echo "Downloading selfback dataset..."
        kaggle datasets download -d ameerkings/human-activity-recognition -p "$OUTPUT_ROOT"
        echo "Unzip selfback dataset..."
        unzip -o -q "$OUTPUT_ROOT"/human-activity-recognition.zip -d "$OUTPUT_ROOT"
        mv "$OUTPUT_ROOT"/selfBACK "$OUTPUT_ROOT"/selfback
        echo "Delete selfback zip file..."
        rm "$OUTPUT_ROOT"/human-activity-recognition.zip
        find "$OUTPUT_ROOT"/selfback -name "._*" -type f -delete
        fi
        ;;
    8)
        echo "Downloading GOTOV dataset"
        curl https://data.4tu.nl/file/f9bae0cd-ec4e-4cfb-aaa5-41bd1c5554ce/ade8856c-6990-439f-9daa-9618db7fb260 -L --output "$OUTPUT_ROOT"/gotov.zip
        echo "Unzip GOTOV dataset..."
        unzip -o "$OUTPUT_ROOT"/gotov.zip -d "$OUTPUT_ROOT"/gotov
        echo "Remove other folders and files except Activity_Measurements data ..."
        mv "$OUTPUT_ROOT"/gotov/Activity_Measurements/* "$OUTPUT_ROOT"/gotov/
        rm -rf "$OUTPUT_ROOT"/gotov/Activity_Measurements
        rm -rf "$OUTPUT_ROOT"/gotov/__MACOSX
        rm "$OUTPUT_ROOT"/gotov.zip
        ;;

    9)
        echo "Downloading from Harvard Dataverse, (HarvardLeo) dataset from study "Daily Living Activity Recognition Using Wearable Devices: A Features-rich Dataset and a Novel Approach""
        python3 $HOME/mp_project/t-st-2023-OneShot-HAR-Hyeongkyun-Orestis/2-Dataset/preprocessing/harvardleo/harvardleo_fetch.py
        ;;
    10)
        echo "Downloading HOUSEHOLDHU dataset"
        curl https://zenodo.org/record/7058383/files/Multimodal_fine_grained_human_activity_data.zip?download=1 -L --output "$OUTPUT_ROOT"/householdhu.zip
        echo "Unzip HOUSEHOLDHU dataset..."
        unzip -o "$OUTPUT_ROOT"/householdhu.zip -d "$OUTPUT_ROOT"/householdhu
        echo "Remove other __MACOSX folder ..."
        rm -rf "$OUTPUT_ROOT"/householdhu/__MACOSX
        rm "$OUTPUT_ROOT"/householdhu.zip
        ;;
        
    11)    
        echo "Downloading MendeleyDaily dataset"
        curl  https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/wjpbtgdyzm-1.zip -L --output "$OUTPUT_ROOT"/mendeleydaily.zip
        echo "Unzip MENDELEYDAILY dataset..."
        unzip -o "$OUTPUT_ROOT"/mendeleydaily.zip -d "$OUTPUT_ROOT"/mendeleydaily
        rm "$OUTPUT_ROOT"/mendeleydaily.zip
        ;;
    12)
        echo "Downloading WristPPG dataset"
        curl https://physionet.org/static/published-projects/wrist/wrist-ppg-during-exercise-1.0.0.zip -L --output "$OUTPUT_ROOT"/wristppg.zip
        echo "Unzip WristPPG dataset..."
        unzip -o "$OUTPUT_ROOT"/wristppg.zip -d "$OUTPUT_ROOT"/wristppg
        echo "Remove zip file ..."
        rm "$OUTPUT_ROOT"/wristppg.zip
        ;;

    13)
        echo "Downloading NewcastleSleep dataset"
        curl https://zenodo.org/record/1160410/files/dataset_psgnewcastle2015_v1.0.zip?download=1 -L --output "$OUTPUT_ROOT"/newcastlesleep.zip
        echo "Unzip NewcastleSleep dataset..."
        unzip -o "$OUTPUT_ROOT"/newcastlesleep.zip -d "$OUTPUT_ROOT"/newcastlesleep
        echo "Remove zip file ..."
        rm "$OUTPUT_ROOT"/newcastlesleep.zip
        ;;
    
    14)
        echo "Downloading Commuting dataset"
        curl https://figshare.com/ndownloader/files/1694495 -L --output "$OUTPUT_ROOT"/commuting.zip
        echo "Unzip Commuting dataset..."
        unzip -o "$OUTPUT_ROOT"/commuting.zip -d "$OUTPUT_ROOT"/commuting
        echo "Remove zip file ..."
        rm "$OUTPUT_ROOT"/commuting.zip
        ;;

    15)
        echo "Downloading Ichi14 dataset"
        curl https://ubicomp.eti.uni-siegen.de/home/datasets/ichi14/ichi14_dataset.zip?lang=en -L --output "$OUTPUT_ROOT"/ichi14.zip
        echo "Unzip Ichi14 dataset..."
        unzip -o "$OUTPUT_ROOT"/ichi14.zip -d "$OUTPUT_ROOT"/ichi14
        echo "Remove zip file ..."
        rm "$OUTPUT_ROOT"/ichi14.zip
        ;;
    16)
        echo "Downloading PAAL dataset"
        curl https://zenodo.org/record/5785955/files/data.zip?download=1 -L --output "$OUTPUT_ROOT"/paal.zip
        echo "Unzip PAAL dataset..."
        unzip -o "$OUTPUT_ROOT"/paal.zip -d "$OUTPUT_ROOT"/paal
        echo "Remove zip file ..."
        rm "$OUTPUT_ROOT"/paal.zip
        ;;
    17)
        if test -f ~/.kaggle/kaggle.json; then
        echo "Downloading *UNSEENDATASET..."
        kaggle datasets download -d kreative24hk/23mp-unseendataset -p "$OUTPUT_ROOT"
        echo "Unzip *UNSEENDATASET"
        unzip -o -q "$OUTPUT_ROOT"/23mp-unseendataset.zip -d "$OUTPUT_ROOT"
        mv "$OUTPUT_ROOT"/unseen_dataset/info.txt "$OUTPUT_ROOT"/unseen_dataset/Willetts2018/info.txt
        mv "$OUTPUT_ROOT"/unseen_dataset/pid.npy "$OUTPUT_ROOT"/unseen_dataset/Willetts2018/pid.npy
        mv "$OUTPUT_ROOT"/unseen_dataset/Y.npy "$OUTPUT_ROOT"/unseen_dataset/Willetts2018/Y.npy
        mv "$OUTPUT_ROOT"/unseen_dataset/X.npy "$OUTPUT_ROOT"/unseen_dataset/Willetts2018/X.npy
        echo "Delete selfback zip file..."
        rm "$OUTPUT_ROOT"/23mp-unseendataset.zip
        fi
        ;;
    *)
        echo "Invalid choice!"
        return 0
        ;;
esac
