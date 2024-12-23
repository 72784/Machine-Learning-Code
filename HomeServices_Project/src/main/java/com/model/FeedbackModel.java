package com.model;

public class FeedbackModel {
private int feedbackid, userid;
private String fullname,subject,email,comment;
public int getFeedbackid() {
	return feedbackid;
}
public void setFeedbackid(int feedbackid) {
	this.feedbackid = feedbackid;
}
public int getUserid() {
	return userid;
}
public void setUserid(int userid) {
	this.userid = userid;
}
public String getFullname() {
	return fullname;
}
public void setFullname(String fullname) {
	this.fullname = fullname;
}
public String getSubject() {
	return subject;
}
public void setSubject(String subject) {
	this.subject = subject;
}
public String getEmail() {
	return email;
}
public void setEmail(String email) {
	this.email = email;
}
public String getComment() {
	return comment;
}
public void setComment(String comment) {
	this.comment = comment;
}
}
