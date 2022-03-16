from app.models.project import Project
from sqlalchemy import func


class SearchService:
    request = None
    query = None
    current_user = None

    def __init__(self, current_user, request):
        self.request = request
        self.query = Project.query
        if current_user.role != 'admin':
            self.current_user = current_user

    def run(self):
        self.filter_by_curent_user()
        self.filter_name()
        return self.query.all()

    def filter_by_curent_user(self):
        if self.current_user:
            self.query = Project.query.filter(Project.company_id==self.current_user.company_id)


    def filter_name(self):
        name = self.request.get('filter[name]')
        if name:
            name = "%{}%".format(name)
            self.query = self.query.filter(func.lower(
                Project.name).like(func.lower(name)))
