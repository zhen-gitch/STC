"""
file       /src/utils/PCGrad.py
"""

import random
import torch


def pcgrad_backward(losses, optimizer, network, param_list=None):
    """
    PCGrad 版本：
    1. 只对共享参数执行 PCGrad 梯度投影
    2. 对任务专属参数执行普通多任务梯度求和
    3. 支持 Lightning 手动优化 network.manual_backward()
    """

    # ============================================================
    # 1. 参数划分
    # ============================================================
    all_params = [
        p for p in network.parameters()
        if p.requires_grad
    ]

    if param_list is None:
        shared_params = all_params
    else:
        shared_params = [
            p for p in param_list
            if p.requires_grad
        ]

    shared_param_ids = {id(p) for p in shared_params}

    non_shared_params = [
        p for p in all_params
        if id(p) not in shared_param_ids
    ]

    num_tasks = len(losses)

    if num_tasks == 0:
        return

    shared_task_grads = []
    non_shared_task_grads = []

    # ============================================================
    # 3. 分别计算每个任务的梯度
    # ============================================================
    for task_idx, loss in enumerate(losses):
        optimizer.zero_grad(set_to_none=True)

        retain = task_idx < num_tasks - 1

        network.manual_backward(
            loss,
            retain_graph=retain
        )

        # 共享参数梯度：展平成一个大向量，用于 PCGrad 投影
        shared_grad_vec = []

        for p in shared_params:
            if p.grad is not None:
                shared_grad_vec.append(
                    p.grad.detach().clone().view(-1)
                )
            else:
                shared_grad_vec.append(
                    torch.zeros_like(p).view(-1)
                )

        if len(shared_grad_vec) > 0:
            shared_task_grads.append(torch.cat(shared_grad_vec))
        else:
            shared_task_grads.append(
                torch.zeros(1, device=loss.device, dtype=loss.dtype)
            )

        # 非共享参数梯度：不做 PCGrad，只保留普通梯度
        current_non_shared_grads = []

        for p in non_shared_params:
            if p.grad is not None:
                current_non_shared_grads.append(
                    p.grad.detach().clone()
                )
            else:
                current_non_shared_grads.append(
                    torch.zeros_like(p)
                )

        non_shared_task_grads.append(current_non_shared_grads)

    # ============================================================
    # 4. 对共享参数执行 PCGrad 投影
    # ============================================================
    original_grads = [
        g.clone()
        for g in shared_task_grads
    ]

    projected_grads = [
        g.clone()
        for g in shared_task_grads
    ]

    for i in range(num_tasks):
        other_task_indices = [
            j for j in range(num_tasks)
            if j != i
        ]

        random.shuffle(other_task_indices)

        for j in other_task_indices:
            g_i = projected_grads[i]
            g_j = original_grads[j]

            dot_product = torch.dot(g_i, g_j)

            if dot_product < 0:
                g_j_norm = torch.dot(g_j, g_j) + 1e-8
                projected_grads[i] = g_i - (dot_product / g_j_norm) * g_j

    total_shared_grad = sum(projected_grads)

    # ============================================================
    # 5. 非共享参数执行普通多任务梯度求和
    # ============================================================
    total_non_shared_grads = []

    for param_idx, p in enumerate(non_shared_params):
        grad_sum = torch.zeros_like(p)

        for task_idx in range(num_tasks):
            grad_sum = grad_sum + non_shared_task_grads[task_idx][param_idx]

        total_non_shared_grads.append(grad_sum)

    # ============================================================
    # 6. 写回梯度
    # ============================================================
    optimizer.zero_grad(set_to_none=True)

    # 写回共享参数梯度
    offset = 0

    for param_idx, p in enumerate(shared_params):
        numel = p.numel()

        if p.grad is None:
            p.grad = torch.zeros_like(p)

        current_grad = total_shared_grad[offset: offset + numel].view_as(p)

        p.grad.copy_(current_grad)

        offset += numel

    # 写回非共享参数梯度
    for param_idx, p in enumerate(non_shared_params):
        if p.grad is None:
            p.grad = torch.zeros_like(p)

        p.grad.copy_(total_non_shared_grads[param_idx])
