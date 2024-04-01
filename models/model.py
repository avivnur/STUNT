from models.protonet_model.mlp import MLPProto


def get_model(P, modelstr):

    if modelstr == 'mlp':
        if 'protonet' in P.mode:
            if P.dataset == 'income':
                model = MLPProto(115, 1024, 1024)
            elif P.dataset == 'cancellation':
                model = MLPProto(23, 1024, 1024)
            elif P.dataset == 'nps':
                model = MLPProto(24, 1024, 1024)
            elif P.dataset == 'drybean':
                model = MLPProto(22, 1024, 1024)
            elif P.dataset == "covtype":
                model = MLPProto(62, 1024, 1024)
            elif P.dataset == "wine":
                model = MLPProto(17, 1024, 1024)

    else:
        raise NotImplementedError()

    return model
