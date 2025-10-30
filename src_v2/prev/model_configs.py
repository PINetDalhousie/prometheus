def configure_model(model, config_file):
    ''' with model object, configure the model using a configuration json file

    Args:
        model: configurable model
        config_file: path to model configuration file
    '''
    # Data to be written
    config_dictionary = {
        "name": "sathiyajith",
        "rollno": 56,
        "cgpa": 8.6,
        "phonenumber": "9976770500"
    }
    
    # Serializing json
    json_object = json.dumps(dictionary, indent=4)
    
    # Writing to sample.json
    with open("sample.json", "w") as outfile:
        outfile.write(json_object)


if __name__ == '__main__':
    pass
