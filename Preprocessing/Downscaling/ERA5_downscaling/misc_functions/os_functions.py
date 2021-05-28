### FOLDER AND FILES:
def check_if_folder_exist_and_create(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    else:
        answer = input("plot directory exists delete (y/n)?: ")
        if answer == "y":
            shutil.rmtree(dir)
            print("deleted folder")
            os.makedirs(dir)
        else:
            exit()
