output "connection_name" {
  value = google_sql_database_instance.this.connection_name
}

output "instance_name" {
  value = google_sql_database_instance.this.name
}

output "database_name" {
  value = google_sql_database.this.name
}

output "user_name" {
  value = google_sql_user.this.name
}

output "password" {
  value     = random_password.db_password.result
  sensitive = true
}
