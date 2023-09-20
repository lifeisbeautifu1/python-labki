from typing import Union, List, Dict
from collections import namedtuple
from datetime import date
import os.path
import json


LAB_WORK_SESSION_KEYS = ("presence", "lab_work_n", "lab_work_mark", "date")
STUDENT_KEYS = ("unique_id", "name", "surname", "group", "subgroup", "lab_works_sessions")


class LabWorkSession(namedtuple('LabWorkSession', 'presence, lab_work_number, lab_work_mark, lab_work_date')):
    startDate = date(23, 9, 1)
    endDate = date(23, 12, 31)
    labs_amount = 6
    """
    Информация о лабораторном занятии, которое могло или не могло быть посещено студентом
    """
    def __new__(cls, presence: bool, lab_work_number: int, lab_work_mark: int, lab_work_date: date):
        """
            param: presence: присутствие студента на л.р.(bool)
            param: lab_work_number: номер л.р.(int)
            param: lab_work_mark: оценка за л.р.(int)
            param: lab_work_date: дата л.р.(date)
        """
        if LabWorkSession._validate_session(
        presence=presence,
        lab_work_number=lab_work_number,
        lab_work_mark=lab_work_mark,
        lab_work_date=lab_work_date):
            return super().__new__(cls, presence=presence,
                                    lab_work_number=lab_work_number,
                                    lab_work_date=lab_work_date,
                                    lab_work_mark=lab_work_mark)

        raise ValueError(f"LabWorkSession ::"
                         f"incorrect args :\n"
                         f"presence       : {presence},\n"
                         f"lab_work_number: {lab_work_number},\n"
                         f"lab_work_mark  : {lab_work_mark},\n"
                         f"lab_work_date  : {lab_work_date}")

    @staticmethod
    def _validate_session(presence: bool, lab_work_number: int, lab_work_mark: int, lab_work_date: date) -> bool:
        """
            param: presence: присутствие студента на л.р.(bool)
            param: lab_work_number: номер л.р.(int)
            param: lab_work_mark: оценка за л.р.(int)
            param: lab_work_date: дата л.р.(date)
        """
        if not (isinstance(presence, bool)):
            return False
        if not (isinstance(lab_work_number, int)):
            return False
        if not (isinstance(lab_work_mark, int)):
            return False
        if not (isinstance(lab_work_date, date)):
            return False
        # if lab_work_mark < 1 or lab_work_mark > 5:
        #     return False
        # if (presence not in (True, False)):
        #     return False
        # if (lab_work_date < LabWorkSession.startDate or lab_work_date > LabWorkSession.endDate):
        #     return False
        # if (lab_work_number < 0 or lab_work_number > LabWorkSession.labs_amount):
        #     return False
        return True

    def __str__(self) -> str:
        """
            Строковое представление LabWorkSession
            Пример:
            {
                    "presence":      1,
                    "lab_work_n":    4,
                    "lab_work_mark": 3,
                    "date":          "15:12:23"
            }
        """
        return f'{{\n\t"presence":\t\t{int(self.presence)},\n\t"lab_work_n":\t\t{self.lab_work_number},\n\t"lab_work_mark":\t{self.lab_work_mark},\n\t"date":\t\t\t"{self.lab_work_date.day}:{self.lab_work_date.month}:{str(self.lab_work_date.year)[len(str(self.lab_work_date.year))-2:]}"\n}}'



class Student:
    __slots__ = ('_unique_id', '_name', '_surname', '_group', '_subgroup', '_lab_work_sessions')

    def __init__(self, unique_id: int, name: str, surname: str, group: int, subgroup: int):
        """
            param: unique_id: уникальный идентификатор студента (int)
            param: name: имя студента (str)
            param: surname: фамилия студента (str)
            param: group: номер группы в которой студент обучается (int)
            param: subgroup: номер подгруппы (int)
        """
        if not self._build_student(unique_id=unique_id,
                                   name=name,
                                   surname=surname,
                                   group=group,
                                   subgroup=subgroup):
            raise ValueError(f"Student ::"
                            f"incorrect args :\n"
                            f"unique_id: {unique_id},\n"
                            f"name:        {name},\n"
                            f"surname: {surname},\n"
                            f"subgroup: {subgroup}")
        self._unique_id = unique_id
        self._name = name
        self._surname = surname
        self._group = group
        self._subgroup = subgroup
        self._lab_work_sessions = []
        

    def _build_student(self, unique_id: int, name: str, surname: str, group: int, subgroup: int) -> bool:
        """
            param: unique_id: уникальный идентификатор студента (int)
            param: name: имя студента (str)
            param: surname: фамилия студента (str)
            param: group: номер группы в которой студент обучается (int)
            param: subgroup: номер подгруппы (int)
        """
        if not (isinstance(unique_id, int)):
            return False
        if not (isinstance(name, str)):
            return False
        if not (isinstance(surname, str)):
            return False
        if not (isinstance(group, int)):
            return False
        if not (isinstance(subgroup, int)):
            return False
        return True

    def __str__(self) -> str:
        """
        Строковое представление Student
        Пример:
        {
                "unique_id":          26,
                "name":               "Щукарев",
                "surname":            "Даниил",
                "group":              6408,
                "subgroup":           2,
                "lab_works_sessions": [
                    {
                        "presence":      1,
                        "lab_work_n":    1,
                        "lab_work_mark": 4,
                        "date":          "15:9:23"
                    },
                    {
                        "presence":      1,
                        "lab_work_n":    2,
                        "lab_work_mark": 4,
                        "date":          "15:10:23"
                    },
                    {
                        "presence":      1,
                        "lab_work_n":    3,
                        "lab_work_mark": 4,
                        "date":          "15:11:23"
                    },
                    {
                        "presence":      1,
                        "lab_work_n":    4,
                        "lab_work_mark": 3,
                        "date":          "15:12:23"
                    }]
        }
        """
        new_line = "\n"
        return f'{{\n\t"unique_id":\t\t{self.unique_id},\n\t"name":\t\t\t"{self.name}",\n\t"surname":\t\t"{self.surname}",\n\t"group":\t\t{self.group},\n\t"subgroup":\t\t{self.subgroup},\n\t"lab_works_sessions":\t[\n{f",{new_line}".join(list(map(lambda sess : f"{str(sess)}", self.lab_work_sessions)))}]\t\n}}'

    @property
    def unique_id(self) -> int:
        """
        Метод доступа для unique_id
        """
        return self._unique_id

    @property
    def group(self) -> int:
        """
        Метод доступа для номера группы
        """
        return self._group

    @property
    def subgroup(self) -> int:
        """
        Метод доступа для номера подгруппы
        """
        return self._subgroup

    @property
    def name(self) -> str:
        """
        Метод доступа для имени студента
        """
        return self._name

    @property
    def surname(self) -> str:
        """
        Метод доступа для фамилии студента
        """
        return self._surname

    @name.setter
    def name(self, val: str) -> None:
        """
        Метод для изменения значения имени студента
        """
        self._name = val

    @surname.setter
    def surname(self, val: str) -> None:
        """
        Метод для изменения значения фамилии студента
        """
        self._surname = val

    @property
    def lab_work_sessions(self):
        """
        Метод доступа для списка лабораторных работ, которые студент посетил или не посетил
        """
        return self._lab_work_sessions

    def append_lab_work_session(self, session: LabWorkSession):
        """
        Метод для регистрации нового лабораторного занятия
        """
        self._lab_work_sessions.append(session)


def _load_lab_work_session(json_node) -> LabWorkSession:
    """
        Создание из под-дерева json файла экземпляра класса LabWorkSession.
        hint: чтобы прочитать дату из формата строки, указанного в json используйте
        date(*tuple(map(int, json_node['date'].split(':'))))
    """
    for key in LAB_WORK_SESSION_KEYS:
        if key not in json_node:
            raise KeyError(f"load_lab_work_session:: key \"{key}\" not present in json_node")
    return ...


def _load_student(json_node) -> Student:
    """
        Создание из под-дерева json файла экземпляра класса Student.
        Если в процессе создания LabWorkSession у студента случается ошибка,
        создание самого студента ломаться не должно.
    """
    for key in STUDENT_KEYS:
        ...
    student = ...
    for session in json_node['lab_works_sessions']:
        ...
    return student


# csv header
#     0    |   1  |   2   |   3  |    4    |  5  |    6    |        7       |       8     |
# unique_id; name; surname; group; subgroup; date; presence; lab_work_number; lab_work_mark
UNIQUE_ID = 0
STUD_NAME = 1
STUD_SURNAME = 2
STUD_GROUP = 3
STUD_SUBGROUP = 4
LAB_WORK_DATE = 5
STUD_PRESENCE = 6
LAB_WORK_NUMBER = 7
LAB_WORK_MARK = 8


def load_students_csv(file_path: str) -> Union[List[Student], None]:
    # csv header
    #     0    |   1  |   2   |   3  |    4    |  5  |    6    |        7       |       8     |
    # unique_id; name; surname; group; subgroup; date; presence; lab_work_number; lab_work_mark
    assert isinstance(file_path, str)
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'r') as filestream:
        list_of_students = []

        for line in filestream:
            currentStudent = line.split(';')
            list_of_students.append(currentStudent)

        list_of_students = list_of_students[1:]
        list_of_students.sort(key=lambda student: int(student[0]))

        for (index, currentStudent) in enumerate(list_of_students):
            currentStudent[6] = bool(currentStudent[6])
            for key in [0, 3, 4, 7, 8]:
                currentStudent[key] = int(currentStudent[key])
            currentStudent[1] = currentStudent[1][1:-1]
            currentStudent[2] = currentStudent[2][1:-1]
            currentStudent[5] = currentStudent[5][1:-1]
            currentStudent[5] = date(*tuple(map(int, currentStudent[5].split(':')[::-1])))

        students = []
        index = 0
        while index < len(list_of_students):
            currentStudent = list_of_students[index]
            lab_works_sessions = []
            while (index < len(list_of_students) and currentStudent[0] == list_of_students[index][0]):
                lab_works_sessions.append({"lab_work_date": list_of_students[index][5], "presence": list_of_students[index][6], "lab_work_number": list_of_students[index][7], "lab_work_mark": list_of_students[index][8]})
                index += 1
            students.append({"unique_id": currentStudent[0], "name": currentStudent[1], "surname": currentStudent[2], "group": currentStudent[3], "subgroup": currentStudent[4], "lab_works_sessions": lab_works_sessions})
        for (index, currentStudent) in enumerate(students):
            try:
                student = Student(unique_id=currentStudent["unique_id"],
                                name=currentStudent["name"],
                                surname=currentStudent["surname"],
                                group=currentStudent["group"],
                                subgroup=currentStudent["subgroup"])
                for lab in currentStudent["lab_works_sessions"]:
                    try:
                        student.append_lab_work_session(
                            LabWorkSession(presence=lab["presence"],
                                            lab_work_number=lab["lab_work_number"],
                                            lab_work_mark=lab["lab_work_mark"],
                                            lab_work_date=lab["lab_work_date"]))
                    except:
                        print(f"Error creating LabWorkSession: {lab}")
                students[index] = student
            except:
                print(f"Error parsing student: {currentStudent}")
                students[index] = None
        return filter(lambda student: student != None, students)


def load_students_json(file_path: str) -> Union[List[Student], None]:
    """
    Загрузка списка студентов из json файла.
    Ошибка создания экземпляра класса Student не должна приводить к поломке всего чтения.
    """
    ...


def save_students_json(file_path: str, students: List[Student]):
    """
    Запись списка студентов в json файл
    """
    ...


def save_students_csv(file_path: str, students: List[Student]):
    """
    Запись списка студентов в csv файл
    """
    ...


if __name__ == '__main__':
    # Задание на проверку json читалки:
    # 1. прочитать файл "students.json"
    # 2. сохранить прочитанный файл в "saved_students.json"
    # 3. прочитать файл "saved_students.json"
    # Задание на проверку csv читалки:
    # 1.-3. аналогично

    # students = load_students_json('students.json')
    # students = save_students_json('students_saved.json')
    # load_students_json('students_saved.json', students)
    
    students = load_students_csv('students.csv')
    for student in students:
        print(student)
    # map(print, students)
    # students = save_students_csv('students_saved.csv')
    # load_students_csv('students_saved.csv', students)
    
    # for s in students:
        # print(s)
    # lab1 = LabWorkSession(bool(1), 1, 5, date(2023, 10, 10))
    # lab2 = LabWorkSession(lab_work_date=date(2023, 9, 23),
    #       presence=bool(0),
    #       lab_work_number=1,
    #       lab_work_mark=5)
    # print(lab1)
    # print(lab2)
    # stud = Student(unique_id=1,name="Архипова",surname="Дарья",group=6408,subgroup=1)
    # stud.append_lab_work_session(lab1)
    # stud.append_lab_work_session(lab2)
    # stud2 = Student(unique_id=21,
    #   name="Полтораднев",
    #   surname="Алексей",
    #   group=6408,
    #   subgroup=2)
    # lab1 = LabWorkSession(lab_work_date=date(2023, 10, 10),
    #       presence=bool(1),
    #       lab_work_number=1,
    #       lab_work_mark=4)
    # stud2.append_lab_work_session(lab1)
    # print(stud2)
    # print(stud)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/