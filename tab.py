import copy
import os
import time

import pandas as pd

from jmetal.algorithm.multiobjective import NSGAII
from jmetal.core.operator import Crossover, S, R
from jmetal.core.problem import Problem
from jmetal.core.solution import IntegerSolution
from jmetal.util.termination_criterion import StoppingByEvaluations

"""
name,count
liluo,24
kakazi,15
55z,12
xdd,15
zhanan,20
ttt,21
atuo,7
jidi,12
jieni,19
boweixi,8
mocha,11
goudalao,16
deng,22
shiguang,9
erciyuan,18
youshang,21
kelei,10
hualian,9
baobao,8
wanwan,10
jige,15
nico,13
sg,14
migan,8
xiaoguaishou,18
shizhou,8
yueyuge,10
sumu,19
shanhai,18
"""

tab_csv = pd.read_csv("./tab.csv", dtype={"count": int})

name_list = tab_csv["name"].tolist()
DESIRED_COUNTS = tab_csv["count"].tolist()

# ----------------------------
# 参数设置
# ----------------------------
NUM_PEOPLE = len(name_list)  # ← 关键修改！
MAX_ROUNDS = 34  # 最大轮次数
SEATS_PER_ROUND = 12  # 每桌最多12人
MIN_SEATS = 12  # 每桌最少10人

TOTAL_DESIRED = sum(DESIRED_COUNTS)

from abc import ABC, abstractmethod
import numpy as np


class AbstractConstraint(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def is_satisfied(self, solution) -> bool:
        """是否满足约束（硬约束判断）"""
        raise NotImplementedError()

    @abstractmethod
    def violation_degree(self, solution) -> float:
        """违反程度（用于软约束或罚函数，0 表示无违反）"""
        raise NotImplementedError()


class AbstractObjective(ABC):
    def __init__(self, name: str, minimize: bool = True):
        self.name = name
        self.minimize = minimize  # True 表示越小越好

    @abstractmethod
    def evaluate(self, solution) -> float:
        """计算目标值"""
        raise NotImplementedError()


# 约束1：同一桌次不能重复报名
class NoDuplicateInRoundConstraint(AbstractConstraint):
    def __init__(self):
        super().__init__("NoDuplicateInRound")

    def is_satisfied(self, solution) -> bool:
        schedule = solution.schedule  # shape: (R, S)
        for round_people in schedule:
            valid = [p for p in round_people if p != -1]
            if len(valid) != len(set(valid)):
                return False
        return True

    def violation_degree(self, solution) -> float:
        degree = 0.0
        schedule = solution.schedule
        for round_people in schedule:
            valid = [p for p in round_people if p != -1]
            duplicates = len(valid) - len(set(valid))
            degree += duplicates * 1000  # 严重惩罚
        return degree


# 约束2：不能超过原本报名次数
class NotExceedDesiredCountConstraint(AbstractConstraint):
    def __init__(self, desired_counts: list):
        super().__init__("NotExceedDesiredCount")
        self.desired_counts = np.array(desired_counts)
        self.num_people = len(desired_counts)

    def is_satisfied(self, solution) -> bool:
        actual = self._get_actual_counts(solution)
        return np.all(actual <= self.desired_counts)

    def violation_degree(self, solution) -> float:
        actual = self._get_actual_counts(solution)
        excess = np.maximum(actual - self.desired_counts, 0)
        return float(np.sum(excess) * 1000)

    def _get_actual_counts(self, solution) -> np.ndarray:
        schedule = solution.schedule
        num_people = self.num_people
        counts = np.zeros(num_people, dtype=int)
        for p in schedule.flat:
            if 0 <= p < num_people:
                counts[p] += 1
        return counts


# 约束3：每桌人数 ∈ [10, 12]（即最多少1~2人）
class TableSizeConstraint(AbstractConstraint):
    def __init__(self, min_seats: int = 10, max_seats: int = 12):
        super().__init__("TableSize")
        self.min_seats = min_seats
        self.max_seats = max_seats

    def is_satisfied(self, solution) -> bool:
        schedule = solution.schedule
        for round_people in schedule:
            n = np.sum(round_people != -1)
            if n > 0 and (n < self.min_seats or n > self.max_seats):
                return False
        return True

    def violation_degree(self, solution) -> float:
        degree = 0.0
        schedule = solution.schedule
        for round_people in schedule:
            n = np.sum(round_people != -1)
            if n == 0:
                continue
            if n < self.min_seats:
                degree += (self.min_seats - n) * 1000
            elif n > self.max_seats:
                degree += (n - self.max_seats) * 1000
        return degree


# 约束4：每人未安排次数 ≤ 1
class MaxUnscheduledConstraint(AbstractConstraint):
    def __init__(self, desired_counts: list, max_unscheduled: int = 1):
        super().__init__("MaxUnscheduled")
        self.desired_counts = np.array(desired_counts)
        self.max_unscheduled = max_unscheduled

    def is_satisfied(self, solution) -> bool:
        actual = self._get_actual_counts(solution)
        shortfall = np.maximum(self.desired_counts - actual, 0)
        return np.all(shortfall <= self.max_unscheduled)

    def violation_degree(self, solution) -> float:
        actual = self._get_actual_counts(solution)
        shortfall = np.maximum(self.desired_counts - actual, 0)
        violations = np.maximum(shortfall - self.max_unscheduled, 0)
        return float(np.sum(violations) * 1000)

    def _get_actual_counts(self, solution) -> np.ndarray:
        schedule = solution.schedule
        num_people = len(self.desired_counts)
        counts = np.zeros(num_people, dtype=int)
        for p in schedule.flat:
            if 0 <= p < num_people:
                counts[p] += 1
        return counts


# 目标1：最小化总等待数（含未安排惩罚）
class WaitingTimeObjective(AbstractObjective):
    def __init__(self, num_people: int, desired_counts: list, max_rounds: int):
        super().__init__("WaitingTime", minimize=True)
        self.num_people = num_people
        self.desired_counts = desired_counts
        self.max_rounds = max_rounds  # 用于未安排者的惩罚

    def evaluate(self, solution) -> float:
        schedule = solution.schedule
        first = np.full(self.num_people, -1, dtype=int)
        last = np.full(self.num_people, -1, dtype=int)
        actual_count = np.zeros(self.num_people, dtype=int)

        for r, round_people in enumerate(schedule):
            for p in round_people:
                if 0 <= p < self.num_people:
                    if first[p] == -1:
                        first[p] = r
                    last[p] = r
                    actual_count[p] += 1

        total_waiting = 0.0
        for i in range(self.num_people):
            desired = self.desired_counts[i]
            actual = actual_count[i]

            if actual == 0:
                # 完全未安排：惩罚 = desired（本应参与 desired 次，全缺）
                waiting = desired
            else:
                span_length = last[i] - first[i] + 1
                gaps_in_span = span_length - actual  # 已安排区间内的空缺
                unassigned = desired - actual  # 未安排次数
                waiting = gaps_in_span + unassigned  # 总等待 = 空缺 + 未安排

            total_waiting += waiting

        return float(total_waiting)


# 目标2：最小化缺人总数（每缺1人，+1）
class TableFullnessObjective(AbstractObjective):
    def __init__(self, max_seats: int = 12):
        super().__init__("TableFullness", minimize=True)
        self.max_seats = max_seats

    def evaluate(self, solution) -> float:
        schedule = solution.schedule  # shape: (num_rounds, seats_per_round)
        total_deficit = 0

        for round_people in schedule:
            # 统计本轮非空座位数
            num_participants = np.sum(round_people == 1)

            # 仅当本轮有至少一人时，才视为“开了这桌”，需要计算缺人
            if num_participants > 0:
                deficit = self.max_seats - num_participants
                total_deficit += deficit

        return float(total_deficit)


# 目标3：最小化总未安排次数（即总缺额）
class TotalShortfallObjective(AbstractObjective):
    def __init__(self, desired_counts: list):
        super().__init__("TotalShortfall", minimize=True)
        self.desired_counts = desired_counts  # list of int

    def evaluate(self, solution) -> float:
        # 统计每人实际参与次数
        actual_counts = [0] * len(self.desired_counts)
        schedule = solution.schedule  # shape: (num_rounds, seats_per_round)

        for round_people in schedule:
            for p in round_people:
                if 0 <= p < len(actual_counts):
                    actual_counts[p] += 1

        # 计算总缺额
        total_shortfall = 0
        for i, desired in enumerate(self.desired_counts):
            actual = actual_counts[i]
            shortfall = max(0, desired - actual)  # 不应为负，但安全起见
            total_shortfall += shortfall

        return total_shortfall


import heapq
from typing import List


class WeightedGreedyScheduler:
    """
    加权贪心调度器：用于生成高质量初始解。

    支持两种优先级模式：
      - 'simple': 仅按剩余次数排序
      - 'weighted': 按 (剩余次数 × 原始需求) 排序
    """

    def __init__(
            self,
            desired_counts: List[int],
            max_rounds: int = 50,
            seats_per_round: int = 12,
            min_seats: int = 10,
            priority_mode: str = "weighted"  # "simple" 或 "weighted"
    ):
        self.desired_counts = desired_counts
        self.num_people = len(desired_counts)
        self.max_rounds = max_rounds
        self.seats_per_round = seats_per_round
        self.min_seats = min_seats
        self.priority_mode = priority_mode

        assert priority_mode in ("simple", "weighted"), \
            "priority_mode must be 'simple' or 'weighted'"

    def generate_schedule_flat(self) -> List[int]:
        """生成扁平化的调度列表（长度 = max_rounds * seats_per_round）"""
        # 初始化最大堆（用负值模拟）
        heap = []
        for person_id, desired in enumerate(self.desired_counts):
            need = desired  # 初始安排 full count
            if need <= 0:
                continue
            priority = self._compute_priority(need, desired)
            # 堆元素: (priority, person_id, current_need, desired)
            heapq.heappush(heap, (priority, person_id, need, desired))

        schedule_flat = [-1] * (self.max_rounds * self.seats_per_round)
        round_idx = 0

        while heap and round_idx < self.max_rounds:
            batch_size = self.seats_per_round
            candidates = []
            popped_items = []

            # 取出最多 batch_size 人
            while heap and len(candidates) < batch_size:
                item = heapq.heappop(heap)
                popped_items.append(item)
                _, pid, need, desired = item
                candidates.append(pid)

            # 检查是否满足最少人数
            if len(candidates) < self.min_seats:
                # 放回所有
                for item in popped_items:
                    heapq.heappush(heap, item)
                break

            # 填入本轮
            start = round_idx * self.seats_per_round
            for j, pid in enumerate(candidates):
                schedule_flat[start + j] = pid

            # 更新剩余需求，放回还有需求的人
            for _, pid, need, desired in popped_items:
                new_need = need - 1
                if new_need > 0:
                    new_priority = self._compute_priority(new_need, desired)
                    heapq.heappush(heap, (new_priority, pid, new_need, desired))

            round_idx += 1

        return schedule_flat

    def _compute_priority(self, remaining: int, desired: int) -> float:
        """
        计算优先级（越小越优先，因为 heapq 是最小堆）
        返回负值以实现最大堆效果
        """
        if self.priority_mode == "simple":
            return -remaining
        elif self.priority_mode == "weighted":
            return -(remaining * desired)
        else:
            raise ValueError(f"Unknown priority_mode: {self.priority_mode}")


from typing import List


class FixedRoundScheduler:
    """
    固定轮数调度器：将总轮数固定为 target_rounds（如 36），
    在此范围内安排所有参与，使每人尽量连续。
    """

    def __init__(
            self,
            desired_counts: List[int],
            target_rounds: int = 36,
            seats_per_round: int = 12,
            min_seats: int = 10
    ):
        self.desired_counts = desired_counts
        self.num_people = len(desired_counts)
        self.target_rounds = target_rounds
        self.seats_per_round = seats_per_round
        self.min_seats = min_seats

    def generate_schedule_flat(self) -> List[int]:
        # 步骤1: 确定每人实际安排次数（允许缺1次）
        total_capacity = self.target_rounds * self.seats_per_round
        actual_counts = []
        total_needed = 0

        for cnt in self.desired_counts:
            # 初始设为 desired
            actual = cnt
            actual_counts.append(actual)
            total_needed += actual

        # 如果超容，随机减少一些人的次数（最多减1）
        if total_needed > total_capacity:
            deficit = total_needed - total_capacity
            candidates = [i for i, cnt in enumerate(self.desired_counts) if cnt > 1]
            random.shuffle(candidates)
            for i in range(min(deficit, len(candidates))):
                actual_counts[candidates[i]] -= 1

        # 步骤2: 生成所有参与事件 [(person_id, priority)]
        events = []
        for pid, actual in enumerate(actual_counts):
            if actual > 0:
                # 权重 = actual（高需求优先连续）
                for _ in range(actual):
                    events.append(pid)

        random.shuffle(events)  # 初始打乱

        # 步骤3: 分配到 target_rounds 轮，尽量连续
        # 使用贪心：按人分组，连续放置
        from collections import defaultdict
        person_events = defaultdict(list)
        for pid in events:
            person_events[pid].append(pid)

        # 按需求降序排序
        sorted_people = sorted(
            person_events.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )

        # 初始化轮次
        rounds = [[] for _ in range(self.target_rounds)]

        # 贪心放置：对每人，找一个能放下其所有参与的连续区间
        for pid, event_list in sorted_people:
            k = len(event_list)
            placed = False

            # 尝试从前往后找连续 k 个有空位的轮次
            for start in range(self.target_rounds - k + 1):
                can_place = True
                for i in range(k):
                    if len(rounds[start + i]) >= self.seats_per_round:
                        can_place = False
                        break
                if can_place:
                    for i in range(k):
                        rounds[start + i].append(pid)
                    placed = True
                    break

            # 如果找不到连续区间，随机分配
            if not placed:
                for _ in range(k):
                    # 找最空的轮次
                    min_len = min(len(r) for r in rounds)
                    candidates = [i for i, r in enumerate(rounds) if
                                  len(r) == min_len and len(r) < self.seats_per_round]
                    if candidates:
                        r_idx = random.choice(candidates)
                        rounds[r_idx].append(pid)

        # 步骤4: 补齐每轮到至少 min_seats（如果可能）
        # （可选：此处可跳过，由优化阶段处理）

        # 转为扁平列表
        flat = []
        for r in rounds:
            # 填充到 seats_per_round（用 -1）
            padded = r[:self.seats_per_round] + [-1] * (self.seats_per_round - len(r))
            flat.extend(padded)

        # 如果超过 MAX_ROUNDS，截断；否则补齐
        max_vars = MAX_ROUNDS * SEATS_PER_ROUND
        if len(flat) > max_vars:
            flat = flat[:max_vars]
        else:
            flat.extend([-1] * (max_vars - len(flat)))

        return flat


class RoundTableProblem(Problem):
    def number_of_variables(self) -> int:
        pass

    def number_of_objectives(self) -> int:
        return len(self.objectives)

    def number_of_constraints(self) -> int:
        return len(self.constraints)

    def name(self) -> str:
        pass

    def __init__(self):
        super(RoundTableProblem, self).__init__()
        self.number_of_variables = MAX_ROUNDS * SEATS_PER_ROUND
        self.obj_directions = [self.MINIMIZE, self.MINIMIZE]
        self.obj_labels = ['Total Waiting Time', 'Non-Full Tables']

        # 变量边界：每个位置 ∈ [-1, 27]
        self.lower_bound = [-1] * self.number_of_variables
        self.upper_bound = [27] * self.number_of_variables

        self.constraints = [
            NoDuplicateInRoundConstraint(),
            NotExceedDesiredCountConstraint(DESIRED_COUNTS),
            TableSizeConstraint(min_seats=MIN_SEATS, max_seats=SEATS_PER_ROUND),
            MaxUnscheduledConstraint(DESIRED_COUNTS, max_unscheduled=1)
        ]

        self.objectives = [
            WaitingTimeObjective(NUM_PEOPLE, DESIRED_COUNTS, MAX_ROUNDS),
            TableFullnessObjective(max_seats=12),
            TotalShortfallObjective(DESIRED_COUNTS)
        ]

        self.number_of_objectives = len(self.objectives)
        self.number_of_constraints = len(self.constraints)

    def evaluate(self, solution: IntegerSolution) -> IntegerSolution:
        assignment = solution.variables
        # reshape to [MAX_ROUNDS, 12]
        schedule = np.array(assignment).reshape((MAX_ROUNDS, SEATS_PER_ROUND))
        solution.schedule = schedule

        solution.objectives = [objective.evaluate(solution) for objective in self.objectives]
        solution.constraints = [constraint.is_satisfied(solution) for constraint in self.constraints]
        return solution

    def create_solution(self) -> IntegerSolution:
        strategy = random.choice(["simple", "weighted", "fixed_round"])

        if strategy == "fixed_round":
            scheduler = FixedRoundScheduler(
                desired_counts=DESIRED_COUNTS,
                target_rounds=MAX_ROUNDS,  # ← 关键参数
                seats_per_round=SEATS_PER_ROUND,
                min_seats=MIN_SEATS
            )
            schedule_flat = scheduler.generate_schedule_flat()
        else:
            priority_mode = strategy
            scheduler = WeightedGreedyScheduler(
                desired_counts=DESIRED_COUNTS,
                max_rounds=MAX_ROUNDS,
                seats_per_round=SEATS_PER_ROUND,
                min_seats=MIN_SEATS,
                priority_mode=priority_mode  # 可改为 "simple"
            )
            schedule_flat = scheduler.generate_schedule_flat()

        # 创建解
        solution = IntegerSolution(
            lower_bound=self.lower_bound,
            upper_bound=self.upper_bound,
            number_of_objectives=self.number_of_objectives,
            number_of_constraints=self.number_of_constraints
        )
        solution.variables = schedule_flat
        return solution


# ... [你的原有 RoundTableProblem 类和所有代码保持不变] ...

# ======================================================================
# 新增：辅助函数（不改动你任何原有逻辑）
# ======================================================================

def rank_solutions(solutions, weights=None):
    objectives = np.array([s.objectives for s in solutions])
    n_obj = objectives.shape[1]

    if weights is None:
        weights = np.ones(n_obj) / n_obj  # 均匀权重
    else:
        weights = np.array(weights)
        assert len(weights) == n_obj, f"weights length {len(weights)} != number of objectives {n_obj}"

    obj_min = objectives.min(axis=0)
    obj_max = objectives.max(axis=0)
    ranges = np.where(obj_max - obj_min == 0, 1.0, obj_max - obj_min)
    normalized = (objectives - obj_min) / ranges
    scores = (normalized * weights).sum(axis=1)
    sorted_indices = np.argsort(scores)
    return sorted_indices, scores, objectives


def export_solution_to_csv(solution, output_path):
    """将单个解导出为 CSV，包含 Round Size 行和统计列"""
    schedule = np.array(solution.variables).reshape((MAX_ROUNDS, SEATS_PER_ROUND))

    # 确定实际轮次数
    actual_rounds = 0
    for r in range(MAX_ROUNDS):
        if any(p != -1 for p in schedule[r]):
            actual_rounds = r + 1
    if actual_rounds == 0:
        return False

    # 构建参与矩阵
    participation = np.full((NUM_PEOPLE, actual_rounds), "", dtype=object)
    for r in range(actual_rounds):
        for p in schedule[r]:
            if 0 <= p < NUM_PEOPLE:
                participation[p, r] = "1"

    round_cols = [f"Round_{i + 1}" for i in range(actual_rounds)]
    df = pd.DataFrame(participation, columns=round_cols)
    df.insert(0, "name", name_list)

    # 添加统计列
    actual_counts = df[round_cols].apply(lambda row: row.str.count("1").sum(), axis=1)
    df["actual_tables"] = actual_counts
    df["desired_tables"] = DESIRED_COUNTS

    # 汇总行：每轮人数 + 总 actual + 总 desired
    round_sizes = [str(int(df[col].str.count("1").sum())) for col in round_cols]
    total_actual = int(actual_counts.sum())
    total_desired = sum(DESIRED_COUNTS)
    summary_row = ["Round Size"] + round_sizes + [str(total_actual), str(total_desired)]
    summary_series = pd.Series(summary_row, index=df.columns)
    df = pd.concat([df, summary_series.to_frame().T], ignore_index=True)

    df.to_csv(output_path, index=False, na_rep="")
    return True


def export_top_solutions(solutions, top_k=10, output_dir="top_solutions", weights=None):
    """导出前 top_k 个解，并生成 summary.csv"""
    sorted_indices, scores, objectives = rank_solutions(solutions, weights)
    top_k = min(top_k, len(solutions))
    os.makedirs(output_dir, exist_ok=True)

    summary_data = []
    for rank in range(top_k):
        idx = sorted_indices[rank]
        sol = solutions[idx]
        filename = os.path.join(output_dir, f"solution_{rank + 1:02d}.csv")

        if export_solution_to_csv(sol, filename):
            summary_data.append({
                "rank": rank + 1,
                "score": scores[idx],
                "total_waiting_time": objectives[idx][0],
                "non_full_tables": objectives[idx][1],
                "non_full_tables2": objectives[idx][2],
                "file": os.path.basename(filename)
            })
            print(f"  Saved solution {rank + 1:2d}: score={scores[idx]:.4f} → {filename}")
        else:
            print(f"  Skipped invalid solution {rank + 1}")

    # 保存 summary
    if summary_data:
        pd.DataFrame(summary_data).to_csv(
            os.path.join(output_dir, "summary.csv"), index=False
        )
        print(f"\n📊 Summary saved to: {output_dir}/summary.csv")

    return len(summary_data)


from jmetal.core.operator import Mutation
from jmetal.core.solution import IntegerSolution


def repair_duplicate_in_rounds(solution: IntegerSolution) -> None:
    """修复每轮中的重复人员：每轮每人最多出现一次"""
    arr = np.array(solution.variables).reshape((MAX_ROUNDS, SEATS_PER_ROUND))

    for r in range(MAX_ROUNDS):
        seen = set()
        for s in range(SEATS_PER_ROUND):
            p = arr[r, s]
            if p == -1:
                continue
            if p in seen:
                # 重复！设为 -1
                arr[r, s] = -1
            else:
                seen.add(p)

    solution.variables = arr.flatten().tolist()


class CompositeScheduleMutation(Mutation):
    """
    组合变异算子：每次随机选择一个子算子执行。
    """

    def get_name(self) -> str:
        return "CompositeScheduleMutation"

    def __init__(self, operators: list, probabilities: list = None):
        """
        :param operators: 子变异算子列表，每个必须是 Mutation 的子类
        :param probabilities: 每个算子被选中的概率（可选，若为 None 则均匀分布）
        """
        super().__init__(probability=1.0)  # 外层概率由 NSGAII 控制，这里设为 1.0
        self.operators = operators
        if probabilities is None:
            n = len(operators)
            self.probabilities = [1.0 / n] * n
        else:
            assert len(probabilities) == len(operators), "Length mismatch"
            total = sum(probabilities)
            self.probabilities = [p / total for p in probabilities]  # 归一化

    def execute(self, solution: IntegerSolution) -> IntegerSolution:
        # 随机选择一个子算子
        chosen_op = random.choices(self.operators, weights=self.probabilities, k=1)[0]
        offspring = chosen_op.execute(solution)

        # 🔧 关键修复：去除每轮中的重复人员
        repair_duplicate_in_rounds(offspring)

        return offspring


class ScheduleOrderMutation(Mutation):
    def get_name(self) -> str:
        return "ScheduleOrderMutation"

    def __init__(self, probability: float = 0.5):
        super().__init__(probability=probability)

    def execute(self, solution: IntegerSolution) -> IntegerSolution:
        if random.random() > self.probability:
            return solution

        # 获取当前调度
        original_vars = solution.variables
        schedule = np.array(original_vars).reshape((MAX_ROUNDS, SEATS_PER_ROUND))

        # 提取非空轮次
        non_empty_rounds = []
        empty_indices = []
        for r in range(MAX_ROUNDS):
            if any(p != -1 for p in schedule[r]):
                non_empty_rounds.append(schedule[r].copy())
            else:
                empty_indices.append(r)

        if len(non_empty_rounds) < 2:
            return solution  # 无法变异

        # === 变异策略：尝试压缩等待时间 ===
        # 策略：按“最早出现的人”对轮次排序（启发式）
        # 更简单：随机打乱非空轮次，然后选一个较好的排列

        # 生成几个候选排列，选目标1最小的
        best_order = non_empty_rounds.copy()
        best_wait = self._compute_waiting_time(best_order)

        # 尝试 K 次随机扰动
        K = 5
        for _ in range(K):
            candidate = non_empty_rounds.copy()
            # 扰动方式1: 随机交换两轮
            i, j = random.sample(range(len(candidate)), 2)
            candidate[i], candidate[j] = candidate[j], candidate[i]

            wait = self._compute_waiting_time(candidate)
            if wait < best_wait:
                best_wait = wait
                best_order = candidate

        # 重建调度表
        new_schedule = np.full((MAX_ROUNDS, SEATS_PER_ROUND), -1, dtype=int)
        for idx, round_data in enumerate(best_order):
            if idx < MAX_ROUNDS:
                new_schedule[idx] = round_data

        # 填回 solution
        solution.variables = new_schedule.flatten().tolist()
        return solution

    def _compute_waiting_time(self, rounds_list):
        """计算给定轮次列表的总等待时间"""
        num_people = NUM_PEOPLE
        first = [None] * num_people
        last = [None] * num_people

        for r_idx, round_people in enumerate(rounds_list):
            for p in round_people:
                if 0 <= p < num_people:
                    if first[p] is None:
                        first[p] = r_idx
                    last[p] = r_idx

        total = 0
        for i in range(num_people):
            if first[i] is not None:
                total += (last[i] - first[i] + 1)
            else:
                total += len(rounds_list)  # 未安排惩罚
        return total


class RandomParticipantSwapMutation(Mutation):
    """
    小变异算子：随机选择两个非空位置（即 != -1），交换其中的人员ID。
    保持每轮人数不变，仅改变谁在哪个轮次。
    """

    def get_name(self) -> str:
        return "RandomParticipantSwapMutation"

    def __init__(self, probability: float = 0.3):
        super().__init__(probability=probability)

    def execute(self, solution: IntegerSolution) -> IntegerSolution:
        if random.random() > self.probability:
            return solution

        # 转为 numpy 数组便于操作
        variables = np.array(solution.variables)
        valid_indices = np.where(variables != -1)[0]

        if len(valid_indices) < 2:
            return solution  # 无法交换

        # 随机选择两个不同的有效位置
        idx1, idx2 = random.sample(valid_indices.tolist(), 2)

        # 交换
        variables[idx1], variables[idx2] = variables[idx2], variables[idx1]

        # 写回
        solution.variables = variables.tolist()
        return solution


import numpy as np
from jmetal.core.operator import Mutation
from jmetal.core.solution import IntegerSolution


class LocalSearchCompactMutation(Mutation):
    """
    局部搜索变异：针对一人，通过多次合法交换，
    尽可能压缩其参与轮次 span（减少等待时间）。

    每次交换保证：
      - 每轮保持12人
      - 无重复人员
    """

    def get_name(self) -> str:
        return "LocalSearchCompactMutation"

    def __init__(self, probability: float = 0.3, max_iterations: int = 10):
        super().__init__(probability=probability)
        self.max_iterations = max_iterations

    def execute(self, solution: IntegerSolution) -> IntegerSolution:
        if random.random() > self.probability:
            return solution

        arr = np.array(solution.variables).reshape((MAX_ROUNDS, SEATS_PER_ROUND))

        # 构建 person -> rounds 集合（用于快速查询）
        person_to_rounds = {}
        for r in range(MAX_ROUNDS):
            for p in arr[r]:
                if p not in person_to_rounds:
                    person_to_rounds[p] = set()
                person_to_rounds[p].add(r)

        if not person_to_rounds:
            return solution

        # 随机选一个人 p
        p = random.choice(list(person_to_rounds.keys()))
        original_rounds = sorted(person_to_rounds[p])
        if len(original_rounds) < 2:
            return solution

        improved = True
        iterations = 0

        while improved and iterations < self.max_iterations:
            improved = False
            current_rounds = sorted(self._get_person_rounds(arr, p))
            if len(current_rounds) < 2:
                break

            first_r, last_r = current_rounds[0], current_rounds[-1]
            span = last_r - first_r + 1
            if span == len(current_rounds):
                break  # 已连续

            # 在 [first_r, last_r] 中找一个 gap 轮次
            gap_rounds = [r for r in range(first_r, last_r + 1) if r not in current_rounds]
            if not gap_rounds:
                break

            rg = random.choice(gap_rounds)
            candidates_in_rg = arr[rg].tolist()

            # 尝试找一个可交换的 q
            for q in candidates_in_rg:
                if q == p:
                    continue
                # 检查 q 是否出现在 p 的任何一轮中
                q_conflict = False
                for pr in current_rounds:
                    if q in arr[pr]:
                        q_conflict = True
                        break
                if q_conflict:
                    continue

                # 找 p 的一个边缘轮次（选最远的以压缩 span）
                # 优先选 last_r 或 first_r
                src_r = last_r if (rg - first_r) < (last_r - rg) else first_r
                if src_r not in current_rounds:
                    continue

                # 执行交换
                try:
                    src_idx = np.where(arr[src_r] == p)[0][0]
                    dst_idx = np.where(arr[rg] == q)[0][0]
                    arr[src_r, src_idx], arr[rg, dst_idx] = q, p
                    improved = True
                    break  # 一次成功交换后重新评估
                except IndexError:
                    continue  # 安全防护

            iterations += 1

        solution.variables = arr.flatten().tolist()
        return solution

    def _get_person_rounds(self, arr: np.ndarray, person: int) -> list:
        """获取某人当前参与的所有轮次"""
        rounds = []
        for r in range(arr.shape[0]):
            if person in arr[r]:
                rounds.append(r)
        return rounds


import random
import numpy as np
from jmetal.core.operator import Mutation
from jmetal.core.solution import IntegerSolution


class BatchFillUnassignedMutation(Mutation):
    """
    批量补位变异：持续利用所有空位，为未安排满的客人补位，
    优先补缺得多的，且位置尽量减少等待时间。
    """

    def get_name(self) -> str:
        return "BatchFillUnassignedMutation"

    def __init__(self, probability: float = 0.5):
        super().__init__(probability=probability)

    def execute(self, solution: IntegerSolution) -> IntegerSolution:
        if random.random() > self.probability:
            return solution

        arr = np.array(solution.variables).reshape((MAX_ROUNDS, SEATS_PER_ROUND))

        # 1. 统计当前状态
        actual_counts = [0] * len(DESIRED_COUNTS)
        person_rounds = [[] for _ in range(len(DESIRED_COUNTS))]
        empty_positions = []

        for r in range(MAX_ROUNDS):
            for s in range(SEATS_PER_ROUND):
                p = arr[r, s]
                if p == -1:
                    empty_positions.append((r, s))
                else:
                    actual_counts[p] += 1
                    person_rounds[p].append(r)

        if not empty_positions:
            return solution  # 无空位

        # 2. 构建可补客人列表 (shortfall, pid)
        candidates = []
        for pid, desired in enumerate(DESIRED_COUNTS):
            actual = actual_counts[pid]
            if actual < desired:
                shortfall = desired - actual
                candidates.append((shortfall, pid))

        if not candidates:
            return solution  # 无人可补

        # 按缺额降序排序
        candidates.sort(key=lambda x: x[0], reverse=True)

        # 3. 对每个空位，尝试分配最优客人
        for (r, s) in empty_positions:
            best_pid = None
            best_score = -1  # 越高越好（如 span 缩短越多）

            for _, pid in candidates:
                # 跳过已在该轮的人
                if pid in arr[r]:
                    continue

                current_rounds = sorted(person_rounds[pid])
                if not current_rounds:
                    # 从未安排：任意位置都一样
                    best_pid = pid
                    break
                else:
                    first_r, last_r = current_rounds[0], current_rounds[-1]
                    old_span = last_r - first_r + 1
                    new_first = min(first_r, r)
                    new_last = max(last_r, r)
                    new_span = new_last - new_first + 1
                    gap_reduction = old_span - (new_span - 1)  # 粗略评分

                    # 更简单：如果 r 在 [first_r, last_r] 内，则 span 不变 → 最优
                    if first_r <= r <= last_r:
                        best_pid = pid
                        break  # 最优，直接选
                    elif gap_reduction > best_score:
                        best_score = gap_reduction
                        best_pid = pid

            if best_pid is not None:
                # 执行分配
                arr[r, s] = best_pid
                actual_counts[best_pid] += 1
                person_rounds[best_pid].append(r)

                # 更新 candidates（移除已满的）
                if actual_counts[best_pid] >= DESIRED_COUNTS[best_pid]:
                    candidates = [(sh, pid) for sh, pid in candidates if pid != best_pid]

                if not candidates:
                    break  # 无人可补

        solution.variables = arr.flatten().tolist()
        return solution


class MyCrossover(Crossover):

    def execute(self, source: S) -> R:
        return source

    def get_name(self) -> str:
        pass

    def get_number_of_parents(self) -> int:
        return 2

    def get_number_of_children(self) -> int:
        return 2


class RoundTableNSGAII(NSGAII):

    def create_initial_solutions(self) -> List[S]:
        visit = set()
        population = []
        for _ in range(self.population_size):
            solution = self.population_generator.new(self.problem)
            repair_duplicate_in_rounds(solution)
            if str(solution.variables) in visit:
                continue
            visit.add(str(solution.variables))
            population.append(solution)
        return population

    def selection(self, population: List[S]) -> List[S]:
        return population

    def reproduction(self, mating_population: List[S]) -> List[S]:
        visit = set()

        for solution in self.solutions:
            visit.add(str(solution.variables))

        offspring_population = []
        for e in mating_population:
            offspring1 = self.mutation_operator.execute(copy.deepcopy(e))
            if str(offspring1.variables) not in visit:
                visit.add(str(offspring1.variables))
                offspring_population.append(offspring1)
        return offspring_population

    def replacement(self, population: List[S], offspring_population: List[S]) -> List[List[S]]:
        return population + offspring_population

    def run(self):
        """Execute the algorithm."""
        self.start_computing_time = time.time()

        self.solutions = self.create_initial_solutions()

        self.solutions = self.evaluate(self.solutions)

        self.init_progress()

        current_steps = 0

        while not self.stopping_condition_is_met():
            self.step()
            self.update_progress()
            current_steps += 1
            if current_steps >= 100000:
                break
            self.total_computing_time = time.time() - self.start_computing_time
            if self.total_computing_time >= 60:
                break

        self.total_computing_time = time.time() - self.start_computing_time


# ======================================================================
# 简洁的 main 函数
# ======================================================================
if __name__ == '__main__':
    # 1. 运行算法（完全保留你的原始写法）
    problem = RoundTableProblem()
    algorithm = RoundTableNSGAII(
        problem=problem,
        population_size=100,
        offspring_population_size=100,
        mutation=CompositeScheduleMutation(
            operators=[ScheduleOrderMutation(), RandomParticipantSwapMutation(), LocalSearchCompactMutation(),
                       BatchFillUnassignedMutation()]),
        crossover=MyCrossover(probability=0.0),
        termination_criterion=StoppingByEvaluations(max_evaluations=10000)
    )
    algorithm.run()
    solutions = algorithm.result()

    if not solutions:
        raise ValueError("No solution found!")
    print(f"✅ Found {len(solutions)} non-dominated solutions.")

    # 2. 导出前10解
    exported = export_top_solutions(solutions, top_k=10, weights=(0.2, 0.2, 0.6))
    print(f"\n🎉 Done! Exported {exported} solutions.")
