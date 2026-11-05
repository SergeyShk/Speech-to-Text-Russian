from bisect import bisect_left, bisect_right
from collections import defaultdict
from collections.abc import Iterable
from itertools import accumulate

from .diarizer import Turn


class Timeline:
    """
    Поиск говорящего по времени

    Аргументы:
        turns (list[Turn]): Реплики записи

    Пример использования:
        >>> from stt_ru.diarization import Timeline, Turn
        >>> timeline = Timeline([Turn(0.0, 2.0, 1), Turn(2.0, 5.0, 2)])
        >>> timeline.speaker(1.0, 3.5), timeline.speaker(6.0, 7.0)
        (2, 2)
    """

    def __init__(self, turns: Iterable[Turn]) -> None:
        self.turns = sorted(turns, key=lambda turn: turn.start)
        self._starts = [turn.start for turn in self.turns]
        self._reach = list(accumulate((turn.end for turn in self.turns), max))
        # Реплика может лежать внутри более длинной: последняя по списку не обязательно ближе
        latest = 0
        self._latest: list[int] = []
        for index, turn in enumerate(self.turns):
            if turn.end >= self.turns[latest].end:
                latest = index
            self._latest.append(latest)

    def speaker(self, start: float, end: float) -> int | None:
        """
        Говорящий на отрезке

        Описание:
            Тот, кто говорит на отрезке дольше всех; если никто - тот, чья реплика ближе всего

        Аргументы:
            start (float): Начало, с
            end (float): Конец, с

        Вывод:
            int: Номер говорящего; None, если реплик нет
        """
        if not self.turns:
            return None
        low = bisect_right(self._reach, start)
        high = bisect_left(self._starts, end)
        spoken: defaultdict[int, float] = defaultdict(float)
        for turn in self.turns[low:high]:
            spoken[turn.speaker] += min(end, turn.end) - max(start, turn.start)
        if spoken:
            return max(spoken, key=lambda speaker: spoken[speaker])
        before = self.turns[self._latest[low - 1]] if low else None
        after = self.turns[high] if high < len(self.turns) else None
        if before is None:
            return None if after is None else after.speaker
        if after is None or start - before.end <= after.start - end:
            return before.speaker
        return after.speaker
