input_file = "movie/task.csv"
output_file = "task_fixed.csv"

with open(input_file, "r", encoding="utf-8") as f:
    content = f.read()

content = content.replace("NULL", "0")

with open(output_file, "w", encoding="utf-8") as f:
    f.write(content)

print("Gotowe! Zapisano:", output_file)