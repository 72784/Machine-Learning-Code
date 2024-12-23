<%@page import="java.sql.ResultSet"%>
<%@page import="java.sql.PreparedStatement"%>
<%@page import="com.util.DBUtil"%>
<%@page import="java.sql.Connection"%>
<%@ page language="java" contentType="text/html; charset=ISO-8859-1"
    pageEncoding="ISO-8859-1"%>
<!DOCTYPE html>
<html>
<head>
<meta charset="ISO-8859-1">
<title>Insert title here</title>
<!-- Favicon icon -->
    <link rel="icon" type="image/png" sizes="16x16" href="images/favicon.png">
    <!-- Pignose Calender -->
    <link href="./plugins/pg-calendar/css/pignose.calendar.min.css" rel="stylesheet">
    <!-- Chartist -->
    <link rel="stylesheet" href="./plugins/chartist/css/chartist.min.css">
    <link rel="stylesheet" href="./plugins/chartist-plugin-tooltips/css/chartist-plugin-tooltip.css">
    <!-- Custom Stylesheet -->
    <link href="css/style.css" rel="stylesheet">
</head>
<body>
   	<jsp:include page="header_sidebar.jsp" />

   <div class="content-body">
<div class="card-body">
                                <div class="active-member">
                                    <div class="card-body">
                                <h4 class="card-title">Company Assignment</h4>
                                <div class="basic-form">
                                
                                
                                
                                
                             
                                    <form action="../AdminController" method="post">
                                        <div class="form-group">
                                            <select class="form-control" name="servicemen">
                                                <option class="hidden" selected="" disabled="">Select Service Company</option>
                                               <%
                                             Connection cn1=new DBUtil().getDatabaseConnection();
                                             PreparedStatement st1 = cn1.prepareStatement("select * from mainservice where serviceid=?");
                                             st1.setString(1, request.getParameter("serviceid"));
                                         ResultSet rs1=st1.executeQuery();
                                         while(rs1.next()){
                                              
                                        
                                                 %>
                                                 <%
                                             Connection cn=new DBUtil().getDatabaseConnection();
                                             PreparedStatement st = cn.prepareStatement("select * from employee where specialization=?");
                                             st.setString(1, rs1.getString(2));
                                         ResultSet rs=st.executeQuery();
                                         while(rs.next())
                                         {
                                             %>
                                                <option value="<%=rs.getString("firstname")%> <%=rs.getString("lastname")%>"><%=rs.getString("firstname")%> <%=rs.getString("lastname")%></option>
                                                <%
                                         }
                                         cn.close();
                                         
				%>
				<%
				}
				                           cn1.close();
				%>
                                            </select>
                                      
                                          </div>
                                        <input type="hidden" name="bookid" value="<%=request.getParameter("bookid")%>">
                                        <input type="hidden" name="servicemen" value="<%=request.getParameter("firstname")%> + +<%= request.getParameter("lastname")%>">
                                        
                                      
                                        <h1>Book id</h1>
                                        <h2>"<%=request.getParameter("bookid")%>"</h2>
                                        <button type="submit" name="action" value="request" class="btn btn-lg btn-outline-primary">Send Request</button>
                                    </form>
                                </div>
                            </div>
                                </div>
                            </div>
                       

</div>
<jsp:include page="footer.jsp" />
 <!--**********************************
        Scripts
    ***********************************-->
    <script src="plugins/common/common.min.js"></script>
    <script src="js/custom.min.js"></script>
    <script src="js/settings.js"></script>
    <script src="js/gleek.js"></script>
    <script src="js/styleSwitcher.js"></script>

    <!-- Chartjs -->
    <script src="./plugins/chart.js/Chart.bundle.min.js"></script>
    <!-- Circle progress -->
    <script src="./plugins/circle-progress/circle-progress.min.js"></script>
    <!-- Datamap -->
    <script src="./plugins/d3v3/index.js"></script>
    <script src="./plugins/topojson/topojson.min.js"></script>
    <script src="./plugins/datamaps/datamaps.world.min.js"></script>
    <!-- Morrisjs -->
    <script src="./plugins/raphael/raphael.min.js"></script>
    <script src="./plugins/morris/morris.min.js"></script>
    <!-- Pignose Calender -->
    <script src="./plugins/moment/moment.min.js"></script>
    <script src="./plugins/pg-calendar/js/pignose.calendar.min.js"></script>
    <!-- ChartistJS -->
    <script src="./plugins/chartist/js/chartist.min.js"></script>
    <script src="./plugins/chartist-plugin-tooltips/js/chartist-plugin-tooltip.min.js"></script>



    <script src="./js/dashboard/dashboard-1.js"></script>


</body>
</html>