from scipy import io


class Biometrics_3():
    def get_xy(self, data):
        _bool = data.corpus.apply(
            lambda x: x in ['EMVIC2014', 'Cerf2007-FIFA', 'ETRA2019'])
        data = data[_bool]
        data = data[~data['subj'].str.contains('test')]
        return data.z, data.subj


class Biometrics_All():
    def get_xy(self, data):
        data.subj = data.subj.apply(lambda x: x.replace('test-', ''))
        return data.z, data.subj


class Biometrics_EMVIC():
    def get_xy(self, data):
        data = data[data.corpus == 'EMVIC2014']
        data = data[~data.subj.str.contains('test')]
        return data.z, data.subj

    def get_test(self, data):
        data = data[data.corpus == 'EMVIC2014']
        data = data[data.subj.str.contains('test')]
        data.subj = data.subj.apply(lambda x: x.replace('test-', ''))
        return data.z, data.subj


class Biometrics_FIFA():
    def get_xy(self, data):
        data = data[data.corpus == 'Cerf2007-FIFA']
        return data.z, data.subj


class Biometrics_ETRA():
    def get_xy(self, data):
        data = data[data.corpus == 'ETRA2019']
        return data.z, data.subj


class Biometrics_ETRA_Fixation():
    def get_xy(self, data):
        data = data[data.corpus == 'ETRA2019-Fixation']
        return data.z, data.subj


class Biometrics_ETRA_All():
    def get_xy(self, data):
        _bool = data.corpus.apply(lambda x: 'ETRA' in x)
        data = data[_bool]
        return data.z, data.subj


class Biometrics_MIT():
    def get_xy(self, data):
        _bool = data.corpus.apply(lambda x: x.startswith('MIT'))
        data = data[_bool]
        return data.z, data.subj


class Biometrics_MIT_LTP():
    def get_xy(self, data):
        _bool = data.corpus.apply(lambda x: x == 'MIT-LearningToPredict')
        data = data[_bool]
        return data.z, data.subj


class Biometrics_MIT_LR():
    def get_xy(self, data):
        _bool = data.corpus.apply(lambda x: x == 'MIT-LowRes')
        data = data[_bool]
        return data.z, data.subj


class Biometrics_MIT_CVCL():
    def get_xy(self, data):
        _bool = data.corpus.apply(lambda x: x == 'MIT-CVCL')
        data = data[_bool]
        return data.z, data.subj
