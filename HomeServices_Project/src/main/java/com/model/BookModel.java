package com.model;

public class BookModel {
private int bookid, userid, id;
private String bdate, price, status, pstatus,extracost;
public String getExtracost() {
	return extracost;
}
public void setExtracost(String extracost) {
	this.extracost = extracost;
}
public int getBookid() {
	return bookid;
}
public void setBookid(int bookid) {
	this.bookid = bookid;
}
public int getUserid() {
	return userid;
}
public void setUserid(int userid) {
	this.userid = userid;
}
public int getId() {
	return id;
}
public void setId(int id) {
	this.id = id;
}
public String getBdate() {
	return bdate;
}
public void setBdate(String bdate) {
	this.bdate = bdate;
}
public String getPrice() {
	return price;
}
public void setPrice(String price) {
	this.price = price;
}
public String getStatus() {
	return status;
}
public void setStatus(String status) {
	this.status = status;
}
public String getPstatus() {
	return pstatus;
}
public void setPstatus(String pstatus) {
	this.pstatus = pstatus;
}
}
