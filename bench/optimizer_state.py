def prime_optimizer(optimizer, torch) -> None:
    parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
        if parameter.requires_grad
    ]
    originals = [parameter.detach().clone() for parameter in parameters]
    for parameter in parameters:
        parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    with torch.no_grad():
        for parameter, original in zip(parameters, originals):
            parameter.copy_(original)
        for group in optimizer.param_groups:
            if "step" in group:
                group["step"] = 0
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    value.zero_()
                elif isinstance(value, (int, float)):
                    state[key] = type(value)(0)
    optimizer.zero_grad(set_to_none=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
