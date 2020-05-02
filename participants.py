print("List of participants")
print("There are currently {} participants.".format(len(os.listdir("./test"))))
for person in os.listdir("./names"):
	print("-", person.split(".")[0])