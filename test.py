import csv
import random
import string
import os


def generate_large_csv(file_path, target_size_mb=300):
    target_size_bytes = target_size_mb * 1024 * 1024  # 转换为字节
    row_count = 0  # 记录行数

    # 打开文件写入数据
    with open(file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 写入表头
        writer.writerow(['a1', 'b1', 'c1', 'd1'])

        while os.path.getsize(file_path) < target_size_bytes:
            # 生成a1：8-12位随机字母+数字
            a1_length = random.randint(8, 12)
            a1 = ''.join(random.choices(string.ascii_letters + string.digits, k=a1_length))

            # 生成b1：0.001-99999.999的浮点数，保留3位小数
            b1 = round(random.uniform(0.001, 99999.999), 3)

            # 生成c1：固定前缀+随机数字（如cat_12345）
            prefix = random.choice(['cat_', 'dog_', 'item_', 'user_'])
            c1_num = random.randint(1000, 99999)
            c1 = f"{prefix}{c1_num}"

            # 生成d1：-10000.0-10000.0的浮点数，保留2位小数
            d1 = round(random.uniform(-10000.0, 10000.0), 2)

            # 写入一行数据
            writer.writerow([a1, b1, c1, d1])
            row_count += 1

            # 每10万行打印一次进度
            if row_count % 100000 == 0:
                current_size_mb = os.path.getsize(file_path) / (1024 * 1024)
                print(f"已生成 {row_count} 行，当前大小：{current_size_mb:.2f} MB")

    final_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"文件生成完成！路径：{file_path}")
    print(f"总行数：{row_count}，最终大小：{final_size_mb:.2f} MB")


generate_large_csv('test_data.csv', target_size_mb=220)