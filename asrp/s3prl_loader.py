import torch
import torchaudio


class s3prlModel:
    def __init__(self, cpkt, dict_path='dict.txt'):
        from s3prl.downstream.runner import Runner
        model_dict = torch.load(cpkt, map_location='cpu')
        self.args = model_dict['Args']
        self.config = model_dict['Config']
        # patch the config
        self.args.init_ckpt = cpkt
        self.args.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config['downstream_expert']['datarc']["dict_path"] = dict_path
        runner = Runner(self.args, self.config)
        self.upstream = runner._get_upstream()
        self.featurizer = runner._get_featurizer()
        self.downstream = runner._get_downstream()

    def __call__(self, wav_path):
        wav, sr = torchaudio.load(wav_path)
        wav = wav.mean(0).unsqueeze(0)
        with torch.no_grad():
            features = self.upstream.model(wav)
            features = self.featurizer.model(wav, features)
            result = self.downstream.model.inference(features, filenames=[])[0]
            return result


s = s3prlModel('./backup/dev-clean-best.ckpt', './backup/char.dict')
print(s('./backup/sample1.flac'))
#["GOING ALONG SLUSHY COUNTRI ROADS AND SPEAKING TO DAMP ORDIANC IS IN DROFTY'S CHOOL ROMS DAY OF TER DAY FOR FOULT NIGHT HE'LL HAVE TO PUT IN AN APPEARANCE OF SUMONG PLACE OF WUSH IT POND SOME DAY MONING AND HE CAN COME TO USSI MEDIATELY OFTOWAODS"]