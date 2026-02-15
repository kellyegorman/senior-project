CREATE DATABASE IF NOT EXISTS capstone_db;
USE capstone_db;
-- users: parent login information
create table users
(userid varchar(15) not null,
username varchar (12) not null,
email varchar(320) not null check(email like '%_@_%._%'),
password_hash varchar(255) not null,
join_date datetime not null default current_timestamp,
primary key(userid),
unique(username),
unique(email)
);
-- devices: monitored devices where the extension is installed
create table devices
(deviceid varchar(15) not null,
userid varchar(15) not null,
device_name varchar(100) not null,
device_token varchar(100) not null,
paired_at timestamp not null default current_timestamp,
primary key(deviceid),
unique(device_token),
foreign key(userid) references users(userid)
);
-- alert category: alert types
create table alert_category(
categoryid varchar(15) not null,
category_name varchar(15) not null,
category_description text,
primary key(categoryid),
unique(category_name)
);
-- alerts: detected alert events
create table alerts(
alertid varchar(15) not null,
deviceid varchar(15) not null,
categoryid varchar(15) not null,
severity varchar(10) not null check (severity in('watch', 'moderate', 'urgent')),
domain varchar(255),
reason_code varchar(100),
created_at timestamp not null default current_timestamp,
primary key(alertid),
foreign key(deviceid) references devices(deviceid),
foreign key(categoryid) references alert_category(categoryid)
);
-- alert settings: parent customization per category
create table alert_settings(
settingid varchar(15) not null,
userid varchar(15) not null,
categoryid varchar(15) not null,
sensitivity integer not null check (sensitivity between 1 and 10),
enabled boolean not null default true,
primary key(settingid),
foreign key(userid) references users(userid),
foreign key(categoryid) references alert_category(categoryid)
);

-- helpful indexes for dashboard queries
create index idx_devices_userid on devices(userid);
create index idx_alerts_deviceid_created_at on alerts(deviceid, created_at desc);
create index idx_alerts_cateogryid on alerts(categoryid);