source_directory="./"

# 目标目录
destination_directory="./"

# 确保目标目录存在
if [ ! -d "$destination_directory" ]; then
    mkdir -p "$destination_directory"
fi

# 遍历源目录中的所有文件夹
for folder in "$source_directory"/*; do
    if [ -d "$folder" ]; then
        folder_name=$(basename "$folder")
        zip_file_path="$destination_directory/$folder_name.zip"
        
        # 压缩文件夹
        zip -r "$zip_file_path" "$folder"
    fi
done