def calculate_all_subset_outputs_tabular(
        model: nn.Module,
        input: torch.Tensor,
        baseline: torch.Tensor,
        calc_bs: Union[None, int] = None,
):
    device = input.device

    assert len(input.shape) == 1
    n_attributes = input.shape[0]

    masks = torch.BoolTensor(generate_all_masks(n_attributes)).to(device)
    masked_inputs = torch.where(masks, input.expand_as(masks), baseline.expand_as(masks))

    if calc_bs is None:
        calc_bs = masks.shape[0]

    outputs = []
    # for batch_id in tqdm(range(int(np.ceil(masks.shape[0] / calc_bs))), ncols=100, desc="Inferencing"):
    for batch_id in range(int(np.ceil(masks.shape[0] / calc_bs))):
        outputs.append(model(masked_inputs[batch_id * calc_bs:batch_id * calc_bs + calc_bs]))
    outputs = torch.cat(outputs, dim=0)

    return masks, outputs


def get_mask_input_func(grid_width: int):

    def generate_masked_input(image: torch.Tensor, baseline: torch.Tensor, grid_indices_list: List):
        device = image.device
        _, image_channel, image_height, image_width = image.shape
        grid_num_h = int(np.ceil(image_height / grid_width))
        grid_num_w = int(np.ceil(image_width / grid_width))
        grid_num = grid_num_h * grid_num_w

        batch_size = len(grid_indices_list)
        masks = torch.zeros(batch_size, image_channel, grid_num)
        for i in range(batch_size):
            grid_indices = flatten(grid_indices_list[i])
            masks[i, :, list(grid_indices)] = 1

        masks = masks.view(masks.shape[0], image_channel, grid_num_h, grid_num_w)
        masks = F.interpolate(
            masks.clone(),
            size=[grid_width * grid_num_h, grid_width * grid_num_w],
            mode="nearest"
        ).float()
        masks = masks[:, :, :image_height, :image_width].to(device)

        expanded_image = image.expand(batch_size, image_channel, image_height, image_width).clone()
        expanded_baseline = baseline.expand(batch_size, image_channel, image_height, image_width).clone()
        masked_image = expanded_image * masks + expanded_baseline * (1 - masks)

        return masked_image

    return generate_masked_input


def calculate_all_subset_outputs_image(
        model: nn.Module,
        input: torch.Tensor,
        baseline: torch.Tensor,
        grid_width: Union[None, int] = None,
        calc_bs: Union[None, int] = None,
        all_players: Union[None, tuple] = None,
        background: Union[None, tuple] = None,
):
    device = input.device
    if len(input.shape) == 3:
        input = input.unsqueeze(0)
    if len(baseline.shape) == 3:
        baseline = baseline.unsqueeze(0)
    assert len(input.shape) == 4
    assert len(baseline.shape) == 4

    _, image_channel, image_height, image_width = input.shape
    grid_num_h = int(np.ceil(image_height / grid_width))
    grid_num_w = int(np.ceil(image_width / grid_width))
    grid_num = grid_num_h * grid_num_w

    mask_input_fn = get_mask_input_func(grid_width=grid_width)

    if all_players is None:
        n_players = grid_num
        all_players = np.arange(grid_num).astype(int)
        masks = torch.BoolTensor(generate_all_masks(n_players))
        grid_indices_list = []
        for i in range(masks.shape[0]):
            player_mask = masks[i]
            grid_indices_list.append(list(flatten(all_players[player_mask])))
    else:
        n_players = len(all_players)
        if background is None:
            background = []
        all_players = np.array(all_players, dtype=object)
        # print("players:", players)
        # print("background:", background)
        masks = torch.BoolTensor(generate_all_masks(n_players))
        grid_indices_list = []
        for i in range(masks.shape[0]):
            player_mask = masks[i]
            grid_indices_list.append(list(flatten([all_players[player_mask], background])))

    if calc_bs is None:
        calc_bs = masks.shape[0]

    assert len(grid_indices_list) == masks.shape[0]

    outputs = []
    for batch_id in tqdm(range(int(np.ceil(len(grid_indices_list) / calc_bs))), ncols=100, desc="Calc model outputs"):
        grid_indices_batch = grid_indices_list[batch_id * calc_bs:batch_id * calc_bs + calc_bs]
        masked_image_batch = mask_input_fn(image=input, baseline=baseline, grid_indices_list=grid_indices_batch)
        output = model(masked_image_batch).cpu()
        outputs.append(output)
    outputs = torch.cat(outputs, dim=0)
    outputs = outputs.to(device)

    return masks, outputs


def calculate_all_subset_outputs(
        model: nn.Module,
        input: torch.Tensor,
        baseline: torch.Tensor,
        grid_width: Union[None, int] = None,
        calc_bs: Union[None, int] = None,
        all_players: Union[None, tuple] = None,
        background: Union[None, tuple] = None,
) -> (torch.Tensor, torch.Tensor):
    '''
    This function returns the output of all possible subsets of the input
    :param model: the target model
    :param input: a single input vector (for tabular data) ...
    :param baseline: the baseline in each dimension
    :return: masks and the outputs
    '''
    if grid_width is None:  # tabular data
        return calculate_all_subset_outputs_tabular(
            model=model, input=input, baseline=baseline,
            calc_bs=calc_bs
        )
    else:  # image data
        return calculate_all_subset_outputs_image(
            model=model, input=input, baseline=baseline,
            grid_width=grid_width, calc_bs=calc_bs,
            all_players=all_players, background=background,
        )



def calculate_given_subset_outputs_image(
        model: nn.Module,
        input: torch.Tensor,
        baseline: torch.Tensor,
        masks: torch.Tensor,
        grid_width: Union[None, int] = None,
        all_players: Union[None, tuple] = None,
        background: Union[None, tuple] = None,
):
    device = input.device
    bs = masks.shape[0]
    all_players = np.array(all_players, dtype=object)

    mask_input_fn = get_mask_input_func(grid_width=grid_width)

    if background is None:
        background = []

    grid_indices_list = []
    for i in range(bs):
        player_mask = masks[i]
        grid_indices_list.append(list(flatten([background, all_players[player_mask]])))

    masked_inputs = mask_input_fn(image=input, baseline=baseline, grid_indices_list=grid_indices_list)
    outputs = model(masked_inputs)

    return masks, outputs


def calculate_given_subset_outputs(model, input, baseline, masks, grid_width=None, all_players=None, background=None):
    if isinstance(model, nn.Module) and grid_width is not None:
        return calculate_given_subset_outputs_image(
            model=model, input=input, baseline=baseline, masks=masks,
            grid_width=grid_width, all_players=all_players, background=background,
        )
    else:
        raise NotImplementedError(f"Unexpected model type: {type(model)}")

