print("List of participants")

for person in os.listdir("./names"):
	print("-", person.split(".")[0])