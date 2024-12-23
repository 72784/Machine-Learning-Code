<!DOCTYPE html>
<%@page import="java.sql.ResultSet"%>
<%@page import="java.sql.PreparedStatement"%>
<%@page import="com.util.DBUtil"%>
<%@page import="java.sql.Connection"%>
<html lang="en">

<head>
    <meta charset="utf-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <title>Quixlab - Bootstrap Admin Dashboard Template by Themefisher.com</title>
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

             <div class="container-fluid">
                <div class="row">
                    <div class="col-lg-12">
                        <div class="card">
                            <div class="card-body">
                                <h4 class="card-title">Input Size</h4>
                                <p class="text-muted">Set heights using classes like <code>.form-control-lg and .form-control-sm.</code>
                                </p>
                                <div class="basic-form">
                                    <form method="post" action="../AdminController">
                                     <div class="form-group">
                                            <select name="serviceid" class="form-control form-control-lg" style="width: 600px;">
                                                                                        <option selected>Open this select menu</option>
                                            
                                             <%
                                             Connection cn=new DBUtil().getDatabaseConnection();
                                             PreparedStatement st = cn.prepareStatement("select * from mainservice");
                                         ResultSet rs=st.executeQuery();
                                         while(rs.next())
                                         {
                                             %>
                                                <option value="<%=rs.getInt(1)%>"><%=rs.getString(2)%></option>
                                             <%
                                         }
                                         cn.close();
                                             %>   
                                             </select>
                                            <br>
                                        <div class="form-group">
                                            <input name="title" class="form-control form-control-lg" type="text" style="width: 600px; height: 30px" placeholder="Add Sub-Service name" fdprocessedid="u1yvs">
                                        </div>
                                       <div class="form-group">
                                            <input name="description" class="form-control form-control-lg" type="text" style="width: 600px; height: 160px" placeholder="Description" fdprocessedid="u1yvs">
                                        </div>
                                        <div class="form-group">
                                            <input name="cost" class="form-control form-control-lg" type="text" style="width: 600px; height: 30px" placeholder="Cost of service (INR)" fdprocessedid="u1yvs">
                                        </div>
<button type="submit" name="action" value="sub-service" class="btn mb-1 btn-rounded btn-primary btn-lg" fdprocessedid="4u2kq">Add Sub Service</button>                           
         </form>
                                </div>
                            </div>
                        </div>
                    </div>   	
                    </div>   	
                    </div>   	
                    </div>
                    
    <jsp:include page="footer.jsp" />
        
        
    <!--**********************************
        Main wrapper end
    ***********************************-->

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